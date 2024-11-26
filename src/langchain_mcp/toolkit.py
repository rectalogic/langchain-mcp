# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import typing as t
from collections.abc import Callable

import pydantic
import pydantic_core
from langchain_core.tools.base import BaseTool, BaseToolkit, InjectedToolArg, ToolException
from mcp import ClientSession


class MCPToolkit(BaseToolkit):
    """
    MCP server toolkit
    """

    session: ClientSession
    """The MCP session used to obtain the tools"""

    _initialized: bool = False

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    async def get_tools(self) -> list[BaseTool]:
        if not self._initialized:
            await self.session.initialize()
            self._initialized = True

        return [
            MCPTool(
                session=self.session,
                name=tool.name,
                description=tool.description,
                args_schema=create_schema_model(tool.inputSchema),
            )
            # list_tools returns a PaginatedResult, but I don't see a way to pass the cursor to retrieve more tools
            for tool in (await self.session.list_tools()).tools
        ]


def create_schema_model(schema: dict[str, t.Any]) -> type[pydantic.BaseModel]:
    # Create a new model class that returns our JSON schema.
    # LangChain requires a BaseModel class.
    class Schema(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

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

    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        raise NotImplementedError("Must invoke tool asynchronously")

    async def _arun(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        result = await self.session.call_tool(self.name, arguments=kwargs)
        content = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

    @property
    def tool_call_schema(self) -> type[pydantic.BaseModel]:
        return self.args_schema
