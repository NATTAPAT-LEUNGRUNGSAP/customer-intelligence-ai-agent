"""Print the canonical MCP and OpenAI tool definitions for a presentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import TOOL_CONTRACTS, mcp_tool_descriptor, openai_function_tool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=sorted(TOOL_CONTRACTS), nargs="?", default="get_segment_products")
    args = parser.parse_args()
    print("MCP tools/list descriptor")
    print(json.dumps(mcp_tool_descriptor(args.tool), ensure_ascii=False, indent=2))
    print("\nOpenAI Responses function tool")
    print(json.dumps(openai_function_tool(args.tool), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
