# langchain-mcp

![PyPI - Version](https://img.shields.io/pypi/v/langchain-mcp)

[Model Context Protocol](https://modelcontextprotocol.io) tool calling support in LangChain.

Create a `langchain_mcp.MCPToolkit` with an `mcp.ClientSession`,
then `await toolkit.initialize()` and `toolkit.get_tools()` to get the list of `langchain_core.tools.BaseTool`s.

Example:

https://github.com/rectalogic/langchain-mcp/blob/8fa8445a24755bf91789f52718c32361ed916f46/tests/demo.py#L34-L43

## Demo

You can run the demo against [Groq](https://groq.com/) `llama-3.1-8b-instant`:
```sh-session
$ export GROQ_API_KEY=xxx
$ uv run tests/demo.py "Read and summarize the file ./LICENSE"
Secure MCP Filesystem Server running on stdio
Allowed directories: [ '/users/aw/projects/rectalogic/langchain-mcp' ]
The file ./LICENSE is a MIT License agreement. It states that the software is provided "as is" without warranty and that the authors and copyright holders are not liable for any claims, damages, or other liability arising from the software or its use.
```