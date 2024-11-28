# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import warnings
from collections.abc import Callable

import pydantic
import pydantic_core
import typing_extensions as t
from langchain_core.tools.base import BaseTool, BaseToolkit, ToolException
from mcp import ClientSession, ListToolsResult


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
                args_schema=create_schema_model(tool.inputSchema),
            )
            # list_tools returns a PaginatedResult, but I don't see a way to pass the cursor to retrieve more tools
            for tool in self._tools.tools
        ]


def create_schema_model(schema: dict[str, t.Any]) -> type[pydantic.BaseModel]:
    # Create a new model class that returns our JSON schema.
    # LangChain requires a BaseModel class.
    class Schema(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

        @t.override
        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool = True,
            ref_template: str = pydantic.json_schema.DEFAULT_REF_TEMPLATE,
            schema_generator: type[pydantic.json_schema.GenerateJsonSchema] = pydantic.json_schema.GenerateJsonSchema,
            mode: pydantic.json_schema.JsonSchemaMode = "validation",
        ) -> dict[str, t.Any]:
            return schema

    return Schema


class MCPTool(BaseTool):
    """
    MCP server tool
    """

    session: ClientSession
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True

    @t.override
    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        warnings.warn(
            "Invoke this tool asynchronousely using `ainvoke`. This method exists only to satisfy standard tests.",
            stacklevel=1,
        )
        return asyncio.run(self._arun(*args, **kwargs))

    @t.override
    async def _arun(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
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
