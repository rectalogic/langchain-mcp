# langchain-mcp

[Model Context Protocol](https://modelcontextprotocol.io) tool calling support in LangChain.

Create a `langchain_mcp.MCPToolkit` with an `mcp.ClientSession`,
then `initialize()` it and `get_tools()` to get the list of `langchain_core.tools.BaseTool`s.

Example:

https://github.com/rectalogic/langchain-mcp/blob/8fa8445a24755bf91789f52718c32361ed916f46/tests/demo.py#L34-L43