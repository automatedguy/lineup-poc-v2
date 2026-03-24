"""
Planner Agent — Consumes explorer output (screenshot + QA analysis + network)
and proposes prioritized test cases using qwen3-vl:8b (local via Ollama).
Saves structured test plans alongside the explorer run output.
"""

import asyncio
import base64
import json
import re
from datetime import datetime
from pathlib import Path

import ollama


class PlannerAgent:

    def __init__(
        self,
        output_dir: str = "runs",
        model: str = "qwen3-vl:8b",
        verbose: bool = False,
        max_cases: int = 10,
    ):
        self.model = model
        self.verbose = verbose
        self.max_cases = max_cases
        self.output_dir = Path(output_dir)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def plan(self, explorer_record: dict, scope: str | None = None) -> dict:
        self._log(f"Planning test cases for {explorer_record['url']}...")

        screenshot_b64 = self._load_screenshot(explorer_record.get("screenshot", ""))
        network_summary = self._load_network(explorer_record.get("network", ""))

        self._log(f"Building prompt (scope: {scope or 'auto-infer'})...")
        messages = self._build_prompt(explorer_record, scope, network_summary, screenshot_b64)

        self._log(f"Generating test cases with {self.model}...")
        raw = self._generate(messages)

        self._log("Parsing response...")
        test_cases = self._parse_response(raw)

        self._log(f"Got {len(test_cases)} test cases, saving...")
        result = self._save(explorer_record, test_cases, scope)
        self._log(f"Saved to {result['test_plan']}")
        return result

    async def plan_many(self, explorer_records: list[dict], scope: str | None = None) -> list[dict]:
        results = []
        for record in explorer_records:
            results.append(await self.plan(record, scope))
        return results

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_screenshot(self, screenshot_path: str) -> str:
        path = Path(screenshot_path)
        if not path.exists():
            self._log(f"Warning: screenshot not found at {screenshot_path}")
            return ""
        return base64.b64encode(path.read_bytes()).decode()

    def _load_network(self, network_path: str) -> str:
        path = Path(network_path)
        if not path.exists():
            return "No network data available."

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "No network data available."

        static_ext = {".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".woff", ".woff2", ".ttf", ".ico"}
        api_endpoints = set()
        for entry in data:
            if entry.get("direction") != "request":
                continue
            url = entry.get("url", "")
            if entry.get("resource_type") in ("document", "xhr", "fetch"):
                api_endpoints.add(f"{entry.get('method', 'GET')} {url}")
            elif not any(url.lower().endswith(ext) for ext in static_ext):
                api_endpoints.add(f"{entry.get('method', 'GET')} {url}")

        if not api_endpoints:
            return "No API calls detected (static page)."
        return "API endpoints found:\n" + "\n".join(f"- {ep}" for ep in sorted(api_endpoints))

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, record: dict, scope: str | None, network_summary: str, screenshot_b64: str) -> list[dict]:
        scope_instruction = scope or (
            "Infer the test scope from the page analysis. "
            "Focus on the functionality visible on this page."
        )

        text = (
            f"You are a senior QA test planner. Based on the following page analysis "
            f"and the screenshot, propose test cases.\n\n"
            f"## Page Under Test\n"
            f"URL: {record['url']}\n\n"
            f"## QA Analysis (from visual inspection)\n"
            f"{record.get('tester_analysis', 'No analysis available.')}\n\n"
            f"## Network Activity Summary\n"
            f"{network_summary}\n\n"
            f"## Scope\n"
            f"{scope_instruction}\n\n"
            f"## Priority Criteria\n"
            f"- critical: Core user journeys, authentication, data integrity — if this fails, the feature is broken\n"
            f"- high: Error handling, input validation, accessibility violations — significant quality issues\n"
            f"- medium: Visual consistency, secondary features, minor UX issues\n\n"
            f"## Instructions\n"
            f"1. Propose up to {self.max_cases} test cases, ordered by priority (critical first).\n"
            f"2. For each test case, provide this exact JSON structure:\n"
            f'   {{"id": "TC-001", "title": "...", "priority": "critical|high|medium", '
            f'"category": "functional|visual|accessibility|performance|security", '
            f'"preconditions": "...", "steps": ["step 1", "step 2"], '
            f'"expected_result": "...", "scope_rationale": "..."}}\n'
            f"3. Return ONLY a JSON array of test case objects. No other text.\n"
            f"4. If something is out of scope, do NOT create a test case for it — "
            f"instead mention it briefly in the scope_rationale of the most related test case.\n"
            f"5. Focus on actionable, specific test cases. Each step must reference "
            f"a concrete UI element from the analysis or screenshot."
        )

        message = {"role": "user", "content": text}
        if screenshot_b64:
            message["images"] = [screenshot_b64]

        return [message]

    # ------------------------------------------------------------------
    # LLM generation (local — Ollama + qwen3-vl)
    # ------------------------------------------------------------------

    def _generate(self, messages: list[dict]) -> str:
        if self.verbose:
            content = ""
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
                    print(token)
                    print("  [response] ", end="", flush=True)
                    continue
                if not in_thinking:
                    content += token
                print(token, end="", flush=True)
            print()
            return content

        response = ollama.chat(model=self.model, messages=messages)
        text = response["message"]["content"]
        # Strip <think>...</think> blocks from non-verbose output
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> list[dict]:
        # Strategy 1: direct parse
        try:
            parsed = json.loads(raw.strip())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract JSON array with regex
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Strategy 3: extract from markdown code fences
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as single unparsed entry
        self._log("Warning: could not parse JSON from LLM response, saving raw text")
        return [{"id": "TC-RAW", "title": "Unparsed response", "raw_text": raw}]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, record: dict, test_cases: list[dict], scope: str | None) -> dict:
        # Derive run directory from explorer's screenshot path
        screenshot_path = Path(record.get("screenshot", ""))
        if screenshot_path.exists():
            run_dir = screenshot_path.parent
        else:
            # Fallback: create new directory
            from urllib.parse import urlparse
            parsed = urlparse(record["url"])
            domain = parsed.netloc or parsed.path
            page = parsed.path.strip("/") or "index"
            page = page.replace("/", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.output_dir / domain / page / ts

        run_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        plan_data = {
            "url": record["url"],
            "scope": scope,
            "timestamp": ts,
            "model": self.model,
            "test_cases": test_cases,
        }

        plan_path = run_dir / "test_plan.json"
        plan_path.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

        # Count priorities
        priorities = {}
        for tc in test_cases:
            p = tc.get("priority", "unknown")
            priorities[p] = priorities.get(p, 0) + 1

        summary = {
            "url": record["url"],
            "timestamp": ts,
            "type": "test_plan",
            "scope": scope,
            "test_plan": str(plan_path),
            "test_case_count": len(test_cases),
            "priorities": priorities,
        }

        return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

async def main():
    import sys

    args = sys.argv[1:]
    verbose = "-v" in args or "--verbose" in args
    args_clean = [a for a in args if not a.startswith("-")]

    # Parse --scope value
    scope = None
    for i, a in enumerate(sys.argv[1:]):
        if a in ("--scope", "-s") and i + 1 < len(sys.argv[1:]):
            scope = sys.argv[i + 2]
            break

    # Parse --max-cases value
    max_cases = 10
    for i, a in enumerate(sys.argv[1:]):
        if a in ("--max-cases", "-n") and i + 1 < len(sys.argv[1:]):
            try:
                max_cases = int(sys.argv[i + 2])
            except ValueError:
                pass
            break

    if not args_clean:
        print("Usage: python -m src.agents.planner <run.jsonl> [--scope 'text'] [--max-cases N] [-v]")
        sys.exit(1)

    jsonl_path = Path(args_clean[0])
    if jsonl_path.is_dir():
        jsonl_path = jsonl_path / "run.jsonl"

    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        sys.exit(1)

    # Load explorer records — handle both JSONL (one per line) and pretty-printed JSON
    records = []
    raw = jsonl_path.read_text(encoding="utf-8").strip()
    try:
        parsed = json.loads(raw)
        # Single object or array
        if isinstance(parsed, dict):
            parsed = [parsed]
        for rec in parsed:
            if rec.get("type") != "test_plan":
                records.append(rec)
    except json.JSONDecodeError:
        # True JSONL: one JSON object per line
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "test_plan":
                continue
            records.append(rec)

    if not records:
        print("No explorer records found in run log.")
        sys.exit(1)

    agent = PlannerAgent(verbose=verbose, max_cases=max_cases)
    for rec in records:
        print(f"Planning test cases for {rec['url']} ...")
        result = await agent.plan(rec, scope=scope)
        print(f"  Test plan → {result['test_plan']}")
        print(f"  Cases: {result['test_case_count']} ({result['priorities']})")
        print()


if __name__ == "__main__":
    asyncio.run(main())
