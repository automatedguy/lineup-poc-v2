"""
Explorer Agent — Navigates URLs, captures screenshot + DOM + network,
and uses qwen3-vl:8b (local via Ollama) to describe the page
from the perspective of a QA tester.
Stores everything in a JSONL run log.
"""

import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path

import ollama
from playwright.async_api import async_playwright


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
        self._log(f"Capturing {url} (screenshot, DOM, network)...")
        screenshot, dom, network = await self._capture(url)
        self._log(f"Capture done — {len(network)} requests logged")
        self._log(f"Analyzing screenshot with {self.model}...")
        analysis = self._analyze(screenshot, url)
        self._log("Analysis complete, saving results...")
        record = self._save(url, screenshot, dom, network, analysis)
        self._log(f"Saved to {record['screenshot']}")
        return record

    async def explore_many(self, urls: list[str]) -> list[dict]:
        results = []
        for url in urls:
            results.append(await self.explore(url))
        return results

    # ------------------------------------------------------------------
    # Browser capture
    # ------------------------------------------------------------------

    async def _capture(self, url: str) -> tuple[bytes, str, list[dict]]:
        network_log: list[dict] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            context = await browser.new_context(viewport={"width": 1280, "height": 720})
            page = await context.new_page()

            page.on("request", lambda req: network_log.append({
                "direction": "request",
                "url": req.url,
                "method": req.method,
                "resource_type": req.resource_type,
            }))
            page.on("response", lambda res: network_log.append({
                "direction": "response",
                "url": res.url,
                "status": res.status,
            }))

            await page.goto(url, wait_until="networkidle", timeout=30_000)
            screenshot = await page.screenshot(full_page=True)
            dom = await page.content()
            await browser.close()

        return screenshot, dom, network_log

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
                # qwen3 emits <think>...</think> blocks for reasoning
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
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, url: str, screenshot: bytes, dom: str, network: list[dict], analysis: str) -> dict:
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

        dom_path = run_dir / "dom.html"
        dom_path.write_text(dom, encoding="utf-8")

        net_path = run_dir / "network.json"
        net_path.write_text(json.dumps(network, indent=2), encoding="utf-8")

        record = {
            "url": url,
            "timestamp": ts,
            "screenshot": str(img_path),
            "dom": str(dom_path),
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
