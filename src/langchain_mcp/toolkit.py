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
from mcp.types import EmbeddedResource, ImageContent, TextContent


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
                args_schema=tool.inputSchema,
            )
            # list_tools returns a PaginatedResult, but I don't see a way to pass the cursor to retrieve more tools
            for tool in self._tools.tools
        ]


class MCPTool(BaseTool):
    """
    MCP server tool
    """

    session: ClientSession
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True
    response_format: t.Literal["content", "content_and_artifact"] = "content_and_artifact"

    @t.override
    def _run(self, *args: t.Any, **kwargs: t.Any) -> tuple[str, list[ImageContent | EmbeddedResource]]:
        warnings.warn(
            "Invoke this tool asynchronousely using `ainvoke`. This method exists only to satisfy standard tests.",
            stacklevel=1,
        )
        return asyncio.run(self._arun(*args, **kwargs))

    @t.override
    async def _arun(self, *args: t.Any, **kwargs: t.Any) -> tuple[str, list[ImageContent | EmbeddedResource]]:
        result = await self.session.call_tool(self.name, arguments=kwargs)
        if result.isError:
            raise ToolException(pydantic_core.to_json(result.content).decode())
        text_content = [block for block in result.content if isinstance(block, TextContent)]
        artifacts = [block for block in result.content if not isinstance(block, TextContent)]
        return pydantic_core.to_json(text_content).decode(), artifacts
