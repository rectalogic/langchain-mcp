# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import sys
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar, Union

import pydantic
import pydantic_core
import typing_extensions as t
from langchain_core.tools.base import BaseTool, BaseToolkit, ToolException
from mcp import ClientSession, ListToolsResult
from pydantic import BaseModel, Field, create_model


class MCPToolkit(BaseToolkit):
    """
    MCP server toolkit
    """

    session: ClientSession
    """The MCP session used to obtain the tools"""

    _tools: ListToolsResult | None = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    async def initialize(self) -> None:
        """Initialize the session and retrieve tools list"""
        if self._tools is None:
            await self.session.initialize()
            self._tools = await self.session.list_tools()

    @t.override
    def get_tools(self) -> list[BaseTool]:
        if self._tools is None:
            raise RuntimeError("Must initialize the toolkit first")

        return [
            MCPTool(
                session=self.session,
                name=tool.name,
                description=tool.description or "",
                args_schema=create_model_from_schema(tool.inputSchema, tool.name),
            )
            for tool in self._tools.tools
        ]


# Define type alias for clarity
JsonSchemaType = type[Any]

TYPEMAP: dict[str, JsonSchemaType] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def resolve_ref(root_schema: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a $ref pointer in the schema"""
    if not ref.startswith("#/"):
        raise ValueError(f"Only local references supported: {ref}")

    path = ref.lstrip("#/").split("/")
    current = root_schema

    for part in path:
        if part not in current:
            raise ValueError(f"Could not find {part} in schema. Available keys: {list(current.keys())}")
        current = current[part]

    return current


def get_field_type(root_schema: dict[str, Any], type_def: dict[str, Any]) -> Any:
    """Convert JSON schema type definition to Python/Pydantic type"""
    # Handle non-dict type definitions (like when additionalProperties is a boolean)
    if not isinstance(type_def, dict):
        return Any

    if "$ref" in type_def:
        referenced_schema = resolve_ref(root_schema, type_def["$ref"])
        # Create a forward reference since the model might not exist yet
        return referenced_schema.get("title", "UntitledModel")

    if "enum" in type_def:
        # Create an Enum class for this field
        enum_name = f"Enum_{hash(str(type_def['enum']))}"
        enum_values = {str(v): v for v in type_def["enum"]}
        return Enum(enum_name, enum_values)

    if "anyOf" in type_def:
        types = [get_field_type(root_schema, t) for t in type_def["anyOf"]]
        # Remove None from types list to handle it separately
        types = [t for t in types if t is not type(None)]  # noqa: E721
        if type(None) in [get_field_type(root_schema, t) for t in type_def["anyOf"]]:
            # If None is one of the possible types, make the field optional
            if len(types) == 1:
                return types[0] | type(None)
            return Union[tuple(types + [type(None)])]  # noqa: UP007
        if len(types) == 1:
            return types[0]
        return Union[tuple(types)]  # noqa: UP007

    if "type" not in type_def:
        return Any

    type_name = type_def["type"]
    if type_name == "array":
        if "items" in type_def:
            item_type = get_field_type(root_schema, type_def["items"])
            return list[item_type]  # type: ignore
        return list[Any]

    if type_name == "object":
        if "additionalProperties" in type_def:
            additional_props = type_def["additionalProperties"]
            # Handle case where additionalProperties is a boolean
            if isinstance(additional_props, bool):
                return dict[str, Any]
            # Handle case where additionalProperties is a schema
            value_type = get_field_type(root_schema, additional_props)
            return dict[str, value_type]  # type: ignore
        return dict[str, Any]

    return TYPEMAP.get(type_name, Any)


ModelType = TypeVar("ModelType", bound=BaseModel)


def create_model_from_schema(
    schema: dict[str, Any], name: str, root_schema: dict[str, Any] | None = None, created_models: set[str] | None = None
) -> type[ModelType]:
    """Create a Pydantic model from a JSON schema definition

    Args:
        schema: The schema for this specific model
        name: Name for the model
        root_schema: The complete schema containing all definitions
        created_models: Set to track which models have already been created
    """
    # Initialize tracking of created models
    if created_models is None:
        created_models = set()

    # If root_schema is not provided, use the current schema as root
    if root_schema is None:
        root_schema = schema

    # If we've already created this model, return its class from the module
    if name in created_models:
        return getattr(sys.modules[__name__], name)

    # Add this model to created_models before processing to handle circular references
    created_models.add(name)

    # Create referenced models first if we have definitions
    if "$defs" in root_schema:
        for model_name, model_schema in root_schema["$defs"].items():
            if model_schema.get("type") == "object" and model_name not in created_models:
                create_model_from_schema(model_schema, model_name, root_schema, created_models)

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_schema in properties.items():
        field_type = get_field_type(root_schema, field_schema)
        default = field_schema.get("default", ...)
        if field_name not in required and default is ...:
            field_type = field_type | type(None)
            default = None

        description = field_schema.get("description", "")
        fields[field_name] = (field_type, Field(default=default, description=description))

    model = create_model(name, **fields)  # type: ignore
    # Add model to the module's namespace so it can be referenced
    setattr(sys.modules[__name__], name, model)
    return model


class MCPTool(BaseTool):
    """
    MCP server tool
    """

    session: ClientSession
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True

    @t.override
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "Invoke this tool asynchronousely using `ainvoke`. This method exists only to satisfy standard tests.",
            stacklevel=1,
        )
        return asyncio.run(self._arun(*args, **kwargs))

    @t.override
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        result = await self.session.call_tool(self.name, arguments=kwargs)
        content = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

    @t.override
    @property
    def tool_call_schema(self) -> type[pydantic.BaseModel]:
        assert self.args_schema is not None  # noqa: S101
        return self.args_schema
