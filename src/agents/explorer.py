"""
Explorer Agent — Navigates URLs, captures screenshot + accessibility snapshot + network
via the Playwright MCP server, and uses qwen3-vl:8b (local via Ollama) to describe
the page from the perspective of a QA tester.
Stores everything in a JSONL run log.
"""

import asyncio
import base64
import json
import re
from datetime import datetime
from pathlib import Path

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVER = StdioServerParameters(
    command="npx",
    args=["@playwright/mcp@latest"]
)


class ExplorerAgent:

    def __init__(self, output_dir: str = "runs", model: str = "qwen3-vl:8b", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def explore(self, url: str) -> dict:
        async with stdio_client(MCP_SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._log(f"Capturing {url} (screenshot, snapshot, network)...")
                screenshot, snapshot, network = await self._capture(session, url)
                self._log(f"Capture done — {len(network)} requests logged")

        if screenshot:
            self._log(f"Analyzing screenshot with {self.model}...")
            analysis = self._analyze(screenshot, url)
            self._log("Analysis complete, saving results...")
        else:
            self._log("WARNING: No screenshot captured, skipping vision analysis")
            analysis = "Screenshot capture failed — no visual analysis available."
        record = self._save(url, screenshot, snapshot, network, analysis)
        self._log(f"Saved to {record['screenshot']}")
        return record

    async def explore_many(self, urls: list[str]) -> list[dict]:
        captures = []
        async with stdio_client(MCP_SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for url in urls:
                    self._log(f"Capturing {url} (screenshot, snapshot, network)...")
                    screenshot, snapshot, network = await self._capture(session, url)
                    self._log(f"Capture done — {len(network)} requests logged")
                    captures.append((url, screenshot, snapshot, network))

        results = []
        for url, screenshot, snapshot, network in captures:
            self._log(f"Analyzing screenshot with {self.model}...")
            analysis = self._analyze(screenshot, url)
            self._log("Analysis complete, saving results...")
            record = self._save(url, screenshot, snapshot, network, analysis)
            self._log(f"Saved to {record['screenshot']}")
            results.append(record)
        return results

    # ------------------------------------------------------------------
    # Browser capture via Playwright MCP
    # ------------------------------------------------------------------

    async def _call(self, session: ClientSession, tool: str, args: dict | None = None) -> object:
        """Call an MCP tool and log errors (suppresses known non-fatal errors)."""
        result = await session.call_tool(tool, args or {})
        if result.isError:
            text = ""
            for block in result.content:
                if block.type == "text":
                    text = block.text
            # browser_run_code returns TypeError for screenshot Buffer — non-fatal, file is saved
            if "is not a function" not in text:
                self._log(f"ERROR from {tool}: {text[:300]}")
        return result

    async def _capture(self, session: ClientSession, url: str) -> tuple[bytes, str, list[dict]]:
        # Navigate to the URL
        nav = await self._call(session, "browser_navigate", {"url": url})
        for block in nav.content:
            if block.type == "text":
                self._log(f"Navigate: {block.text[:150]}")

        # Take screenshot (browser_take_screenshot has a 5s timeout, too short for heavy pages)
        tmp_png = f"/tmp/explorer_{id(session)}.png"
        await self._call(session, "browser_run_code", {
            "code": f"await page.screenshot({{ path: '{tmp_png}', fullPage: true, timeout: 30000 }})"
        })
        tmp = Path(tmp_png)
        screenshot = tmp.read_bytes() if tmp.exists() else b""
        if tmp.exists():
            tmp.unlink()
        self._log(f"Screenshot: {len(screenshot)} bytes")

        # Get accessibility snapshot
        snap = await self._call(session, "browser_snapshot")
        snapshot = ""
        for block in snap.content:
            if block.type == "text":
                snapshot = block.text
                break

        # Get network requests — MCP returns "[METHOD] URL" lines
        net = await self._call(session, "browser_network_requests")
        network = []
        for block in net.content:
            if block.type == "text":
                for line in block.text.strip().splitlines():
                    m = re.match(r'\[(\w+)\]\s+(.*)', line)
                    if m:
                        network.append({"method": m.group(1), "url": m.group(2)})
                break

        return screenshot, snapshot, network

    # ------------------------------------------------------------------
    # Vision analysis (local — Ollama + qwen3-vl)
    # ------------------------------------------------------------------

    def _analyze(self, screenshot: bytes, url: str) -> str:
        b64 = base64.b64encode(screenshot).decode()

        messages = [{
            "role": "user",
            "content": (
                f"You are a senior QA tester. Analyse this screenshot of {url}.\n\n"
                "Document the following:\n"
                "1. **Page Layout** — overall structure and visual hierarchy.\n"
                "2. **UI Elements** — buttons, forms, links, navigation, modals, dropdowns.\n"
                "3. **Content** — visible text, images, data tables.\n"
                "4. **Potential Issues** — visual bugs, alignment, broken images, accessibility.\n"
                "5. **Interactive Elements** — what can be clicked, typed, toggled.\n"
                "6. **State** — loading indicators, error messages, empty states.\n\n"
                "Be specific about positions and appearances."
            ),
            "images": [b64],
        }]

        if self.verbose:
            content = ""
            thinking = ""
            in_thinking = False
            for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
                token = chunk["message"]["content"]
                if "<think>" in token:
                    in_thinking = True
                    print("  [thinking] ", end="", flush=True)
                    token = token.replace("<think>", "")
                if "</think>" in token:
                    in_thinking = False
                    token = token.replace("</think>", "")
                    thinking += token
                    print(token)
                    print("  [response] ", end="", flush=True)
                    continue
                if in_thinking:
                    thinking += token
                else:
                    content += token
                print(token, end="", flush=True)
            print()
            return content

        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]

    # ------------------------------------------------------------------
    # Snapshot parsing
    # ------------------------------------------------------------------

    def _parse_snapshot(self, raw: str) -> list[dict]:
        """Parse Playwright accessibility snapshot into structured elements."""
        elements = []
        yaml_match = re.search(r'```yaml\n(.*?)```', raw, re.DOTALL)
        if not yaml_match:
            return elements

        current = None
        for line in yaml_match.group(1).splitlines():
            stripped = line.lstrip()
            if not stripped.startswith('- '):
                continue
            content = stripped[2:]
            depth = (len(line) - len(stripped)) // 2

            # Property lines belong to current element
            if content.startswith('/url:'):
                if current:
                    current['url'] = content[5:].strip()
                continue
            if content.startswith('text:'):
                if current:
                    current['text'] = content[5:].strip()
                continue

            # Parse element: role "name" [ref=eN] [attr=val]
            part = content.rstrip(':').strip()
            role_m = re.match(r'(\w+)', part)
            if not role_m:
                continue

            elem = {"role": role_m.group(1), "depth": depth}
            name_m = re.search(r'"([^"]*)"', part)
            if name_m:
                elem["name"] = name_m.group(1)
            ref_m = re.search(r'\[ref=(\w+)\]', part)
            if ref_m:
                elem["ref"] = ref_m.group(1)
            for attr_m in re.finditer(r'\[(\w+)(?:=([^\]]+))?\]', part):
                if attr_m.group(1) != 'ref':
                    elem.setdefault("attrs", {})[attr_m.group(1)] = attr_m.group(2) or True

            current = elem
            elements.append(elem)

        return elements

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, url: str, screenshot: bytes, snapshot: str, network: list[dict], analysis: str) -> dict:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        page = parsed.path.strip("/") or "index"
        page = page.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = self.output_dir / domain / page / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        img_path = run_dir / "screenshot.png"
        img_path.write_bytes(screenshot)

        snapshot_path = run_dir / "snapshot.json"
        snapshot_data = self._parse_snapshot(snapshot)
        snapshot_path.write_text(json.dumps(snapshot_data, indent=2), encoding="utf-8")

        net_path = run_dir / "network.json"
        net_path.write_text(json.dumps(network, indent=2), encoding="utf-8")

        record = {
            "url": url,
            "timestamp": ts,
            "screenshot": str(img_path),
            "dom": str(snapshot_path),
            "network": str(net_path),
            "tester_analysis": analysis,
        }

        run_log_path = run_dir / "state.jsonl"
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

async def main():
    import sys
    args = sys.argv[1:]
    verbose = "-v" in args or "--verbose" in args
    urls = [a for a in args if not a.startswith("-")] or ["https://example.com"]
    agent = ExplorerAgent(verbose=verbose)
    for url in urls:
        print(f"Exploring {url} ...")
        record = await agent.explore(url)
        print(f"  Screenshot → {record['screenshot']}")
        print(f"  Analysis:\n{record['tester_analysis'][:300]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
