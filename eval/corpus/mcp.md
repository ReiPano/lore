# Model Context Protocol

The Model Context Protocol (MCP) is an open standard for connecting agents to
external tools, data, and context. An MCP server exposes tools (callable
functions), resources (read-only addressable content), and prompts. Claude
Desktop and Claude Code both speak MCP over stdio, so a single server
implementation can plug into either client. FastMCP is the Python SDK for
building MCP servers with minimal boilerplate.
