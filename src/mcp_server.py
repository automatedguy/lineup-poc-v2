"""
Playwright MCP Server — Exposes browser automation tools via MCP protocol.
Designed for the lineup QA test generation pipeline.

Run:
    python src/mcp_server.py

Configure in claude_desktop_config.json or .mcp.json:
    {
      "mcpServers": {
        "playwright": {
          "command": "python",
          "args": ["src/mcp_server.py"]
        }
      }
    }
"""

import base64
import json

from mcp.server.fastmcp import FastMCP, Image
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

mcp = FastMCP("playwright")

# ---------------------------------------------------------------------------
# Browser session state
# ---------------------------------------------------------------------------

_playwright = None
_browser: Browser | None = None
_context: BrowserContext | None = None
_page: Page | None = None
_network_log: list[dict] = []
_console_log: list[dict] = []


async def _ensure_browser() -> Page:
    """Launch browser if needed and return the active page."""
    global _playwright, _browser, _context, _page, _network_log, _console_log

    if _page and not _page.is_closed():
        return _page

    if not _playwright:
        _playwright = await async_playwright().start()

    if not _browser or not _browser.is_connected():
        _browser = await _playwright.chromium.launch(headless=True)

    _network_log = []
    _console_log = []

    _context = await _browser.new_context(viewport={"width": 1280, "height": 720})
    _page = await _context.new_page()

    _page.on("request", lambda req: _network_log.append({
        "direction": "request",
        "url": req.url,
        "method": req.method,
        "resource_type": req.resource_type,
    }))
    _page.on("response", lambda res: _network_log.append({
        "direction": "response",
        "url": res.url,
        "status": res.status,
    }))
    _page.on("console", lambda msg: _console_log.append({
        "type": msg.type,
        "text": msg.text,
    }))

    return _page


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

@mcp.tool()
async def navigate(url: str, wait_until: str = "load") -> str:
    """Navigate to a URL.

    Args:
        url: The URL to navigate to.
        wait_until: When to consider navigation complete —
                    "load", "domcontentloaded", "networkidle", or "commit".
    """
    page = await _ensure_browser()
    global _network_log, _console_log
    _network_log = []
    _console_log = []
    response = await page.goto(url, wait_until=wait_until, timeout=30_000)
    status = response.status if response else "unknown"
    return f"Navigated to {url} (status: {status}, title: {await page.title()})"


@mcp.tool()
async def go_back() -> str:
    """Navigate back in browser history."""
    page = await _ensure_browser()
    response = await page.go_back(timeout=10_000)
    status = response.status if response else "no previous page"
    return f"Went back (status: {status}, url: {page.url})"


@mcp.tool()
async def go_forward() -> str:
    """Navigate forward in browser history."""
    page = await _ensure_browser()
    response = await page.go_forward(timeout=10_000)
    status = response.status if response else "no forward page"
    return f"Went forward (status: {status}, url: {page.url})"


@mcp.tool()
async def reload() -> str:
    """Reload the current page."""
    page = await _ensure_browser()
    global _network_log, _console_log
    _network_log = []
    _console_log = []
    response = await page.reload(timeout=30_000)
    status = response.status if response else "unknown"
    return f"Reloaded (status: {status})"


# ---------------------------------------------------------------------------
# Screenshots & page info
# ---------------------------------------------------------------------------

@mcp.tool()
async def screenshot(selector: str | None = None, full_page: bool = False) -> Image:
    """Take a screenshot of the current page or a specific element.

    Args:
        selector: CSS selector of element to screenshot. Captures viewport if omitted.
        full_page: If True and no selector, capture the full scrollable page.
    """
    page = await _ensure_browser()
    if selector:
        element = await page.query_selector(selector)
        if not element:
            raise ValueError(f"Element not found: '{selector}'")
        img_bytes = await element.screenshot()
    else:
        img_bytes = await page.screenshot(full_page=full_page)
    return Image(data=base64.b64encode(img_bytes).decode(), format="png")


@mcp.tool()
async def get_page_content() -> str:
    """Get the full HTML content of the current page."""
    page = await _ensure_browser()
    return await page.content()


@mcp.tool()
async def get_page_info() -> str:
    """Get current page URL, title, and viewport size."""
    page = await _ensure_browser()
    return json.dumps({
        "url": page.url,
        "title": await page.title(),
        "viewport": page.viewport_size,
    }, indent=2)


@mcp.tool()
async def get_text(selector: str) -> str:
    """Get the text content of an element.

    Args:
        selector: CSS selector of the element.
    """
    page = await _ensure_browser()
    element = await page.query_selector(selector)
    if not element:
        raise ValueError(f"Element not found: '{selector}'")
    return await element.text_content() or ""


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

@mcp.tool()
async def click(selector: str) -> str:
    """Click on an element.

    Args:
        selector: CSS selector of the element to click.
    """
    page = await _ensure_browser()
    await page.click(selector, timeout=10_000)
    return f"Clicked '{selector}'"


@mcp.tool()
async def fill(selector: str, value: str) -> str:
    """Fill a form input with text (clears existing value first).

    Args:
        selector: CSS selector of the input element.
        value: Text to fill in.
    """
    page = await _ensure_browser()
    await page.fill(selector, value, timeout=10_000)
    return f"Filled '{selector}' with '{value}'"


@mcp.tool()
async def type_text(selector: str, text: str, delay: int = 50) -> str:
    """Type text character by character (simulates real keystrokes).

    Args:
        selector: CSS selector of the element to type into.
        text: Text to type.
        delay: Delay between keystrokes in ms.
    """
    page = await _ensure_browser()
    await page.type(selector, text, delay=delay, timeout=10_000)
    return f"Typed into '{selector}'"


@mcp.tool()
async def press_key(key: str, selector: str | None = None) -> str:
    """Press a keyboard key.

    Args:
        key: Key to press (e.g. "Enter", "Tab", "Escape", "ArrowDown").
        selector: Optional CSS selector to focus before pressing.
    """
    page = await _ensure_browser()
    if selector:
        await page.press(selector, key, timeout=10_000)
    else:
        await page.keyboard.press(key)
    return f"Pressed '{key}'" + (f" on '{selector}'" if selector else "")


@mcp.tool()
async def hover(selector: str) -> str:
    """Hover over an element.

    Args:
        selector: CSS selector of the element.
    """
    page = await _ensure_browser()
    await page.hover(selector, timeout=10_000)
    return f"Hovered over '{selector}'"


@mcp.tool()
async def select_option(selector: str, value: str) -> str:
    """Select an option from a <select> dropdown.

    Args:
        selector: CSS selector of the <select> element.
        value: The value attribute of the option to select.
    """
    page = await _ensure_browser()
    selected = await page.select_option(selector, value, timeout=10_000)
    return f"Selected {selected} in '{selector}'"


@mcp.tool()
async def check(selector: str) -> str:
    """Check a checkbox or radio button.

    Args:
        selector: CSS selector of the checkbox/radio.
    """
    page = await _ensure_browser()
    await page.check(selector, timeout=10_000)
    return f"Checked '{selector}'"


@mcp.tool()
async def uncheck(selector: str) -> str:
    """Uncheck a checkbox.

    Args:
        selector: CSS selector of the checkbox.
    """
    page = await _ensure_browser()
    await page.uncheck(selector, timeout=10_000)
    return f"Unchecked '{selector}'"


# ---------------------------------------------------------------------------
# Waiting & JS evaluation
# ---------------------------------------------------------------------------

@mcp.tool()
async def wait_for_selector(
    selector: str, state: str = "visible", timeout: int = 10000
) -> str:
    """Wait for an element to reach a given state.

    Args:
        selector: CSS selector to wait for.
        state: Target state — "attached", "detached", "visible", or "hidden".
        timeout: Maximum wait time in ms.
    """
    page = await _ensure_browser()
    await page.wait_for_selector(selector, state=state, timeout=timeout)
    return f"'{selector}' reached state '{state}'"


@mcp.tool()
async def evaluate(expression: str) -> str:
    """Execute JavaScript in the page and return the result.

    Args:
        expression: JavaScript expression to evaluate.
    """
    page = await _ensure_browser()
    result = await page.evaluate(expression)
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Logs & element discovery
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_network_log() -> str:
    """Return captured network requests/responses since last navigation."""
    return json.dumps(_network_log, indent=2)


@mcp.tool()
async def get_console_log() -> str:
    """Return captured browser console messages since last navigation."""
    return json.dumps(_console_log, indent=2)


@mcp.tool()
async def get_visible_elements(
    selector: str = "a, button, input, select, textarea, [role='button'], [onclick]",
) -> str:
    """List interactive elements visible on the page.

    Args:
        selector: CSS selector to match. Defaults to common interactive elements.

    Returns:
        JSON array of visible elements with tag, text, and key attributes (max 100).
    """
    page = await _ensure_browser()
    elements = await page.evaluate(
        """(selector) => {
            return Array.from(document.querySelectorAll(selector))
                .filter(el => {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    return rect.width > 0 && rect.height > 0
                        && style.display !== 'none'
                        && style.visibility !== 'hidden';
                })
                .slice(0, 100)
                .map(el => ({
                    tag: el.tagName.toLowerCase(),
                    type: el.type || null,
                    text: (el.textContent || '').trim().slice(0, 100),
                    id: el.id || null,
                    name: el.name || null,
                    href: el.href || null,
                    placeholder: el.placeholder || null,
                    ariaLabel: el.getAttribute('aria-label') || null,
                    selector: el.id ? '#' + el.id
                        : el.name ? el.tagName.toLowerCase() + '[name="' + el.name + '"]'
                        : null,
                }));
        }""",
        selector,
    )
    return json.dumps(elements, indent=2)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@mcp.tool()
async def close_browser() -> str:
    """Close the browser and free resources."""
    global _playwright, _browser, _context, _page, _network_log, _console_log
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()
    _playwright = None
    _browser = None
    _context = None
    _page = None
    _network_log = []
    _console_log = []
    return "Browser closed"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
