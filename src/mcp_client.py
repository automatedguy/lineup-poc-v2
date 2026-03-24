"""
MCP Client — Wraps the Playwright MCP server as a reusable async client.
Agents use this instead of calling Playwright directly.
"""

import base64
import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpClient:

    def __init__(self, server_script: str = "src/mcp_server.py"):
        self._server_script = server_script
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._session:
            return

        self._exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self._server_script],
        )
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    async def close(self) -> None:
        if self._session:
            try:
                await self.call("close_browser")
            except Exception:
                pass
        if self._exit_stack:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # ------------------------------------------------------------------
    # Generic call
    # ------------------------------------------------------------------

    async def call(self, tool: str, args: dict | None = None) -> list:
        """Call an MCP tool and return raw content items."""
        if not self._session:
            await self.connect()
        result = await self._session.call_tool(tool, args or {})
        return result.content

    # ------------------------------------------------------------------
    # Typed helpers
    # ------------------------------------------------------------------

    async def navigate(self, url: str, wait_until: str = "networkidle") -> str:
        items = await self.call("navigate", {"url": url, "wait_until": wait_until})
        return items[0].text if items else ""

    async def screenshot(self, *, full_page: bool = False, selector: str | None = None) -> bytes:
        args: dict = {"full_page": full_page}
        if selector:
            args["selector"] = selector
        items = await self.call("screenshot", args)
        for item in items:
            if item.type == "image":
                return base64.b64decode(item.data)
        return b""

    async def get_page_content(self) -> str:
        items = await self.call("get_page_content")
        return items[0].text if items else ""

    async def get_network_log(self) -> list[dict]:
        items = await self.call("get_network_log")
        return json.loads(items[0].text) if items else []

    async def get_console_log(self) -> list[dict]:
        items = await self.call("get_console_log")
        return json.loads(items[0].text) if items else []

    async def get_page_info(self) -> dict:
        items = await self.call("get_page_info")
        return json.loads(items[0].text) if items else {}

    async def click(self, selector: str) -> str:
        items = await self.call("click", {"selector": selector})
        return items[0].text if items else ""

    async def fill(self, selector: str, value: str) -> str:
        items = await self.call("fill", {"selector": selector, "value": value})
        return items[0].text if items else ""

    async def type_text(self, selector: str, text: str, delay: int = 50) -> str:
        items = await self.call("type_text", {"selector": selector, "text": text, "delay": delay})
        return items[0].text if items else ""

    async def press_key(self, key: str, selector: str | None = None) -> str:
        args: dict = {"key": key}
        if selector:
            args["selector"] = selector
        items = await self.call("press_key", args)
        return items[0].text if items else ""

    async def hover(self, selector: str) -> str:
        items = await self.call("hover", {"selector": selector})
        return items[0].text if items else ""

    async def select_option(self, selector: str, value: str) -> str:
        items = await self.call("select_option", {"selector": selector, "value": value})
        return items[0].text if items else ""

    async def check(self, selector: str) -> str:
        items = await self.call("check", {"selector": selector})
        return items[0].text if items else ""

    async def uncheck(self, selector: str) -> str:
        items = await self.call("uncheck", {"selector": selector})
        return items[0].text if items else ""

    async def wait_for_selector(self, selector: str, state: str = "visible", timeout: int = 10000) -> str:
        items = await self.call("wait_for_selector", {"selector": selector, "state": state, "timeout": timeout})
        return items[0].text if items else ""

    async def evaluate(self, expression: str) -> str:
        items = await self.call("evaluate", {"expression": expression})
        return items[0].text if items else ""

    async def get_visible_elements(self, selector: str | None = None) -> list[dict]:
        args: dict = {}
        if selector:
            args["selector"] = selector
        items = await self.call("get_visible_elements", args)
        return json.loads(items[0].text) if items else []

    async def go_back(self) -> str:
        items = await self.call("go_back")
        return items[0].text if items else ""

    async def go_forward(self) -> str:
        items = await self.call("go_forward")
        return items[0].text if items else ""

    async def reload(self) -> str:
        items = await self.call("reload")
        return items[0].text if items else ""
