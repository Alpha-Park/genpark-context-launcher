# -*- coding: utf-8 -*-
"""
GenPark Context Launcher MCP Server
FastMCP standard interface for attention softmax allocation, cache block alignment, and token packing.
"""

import json
from typing import Dict, Any, List
from client import GenParkContextLauncher

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("genpark-context-launcher")
except ImportError:
    class MockFastMCP:
        def __init__(self, name):
            self.name = name
        def tool(self):
            def decorator(f):
                return f
            return decorator
    mcp = MockFastMCP("genpark-context-launcher")

launcher = GenParkContextLauncher()

@mcp.tool()
def allocate_attention_weights_softmax(
    context_items: List[Dict[str, Any]],
    temperature: float = 1.0
) -> str:
    """
    Computes normalized attention allocation weights across context layers using Softmax.
    """
    res = launcher.allocate_attention_weights_softmax(context_items, temperature)
    return json.dumps(res, indent=2)

@mcp.tool()
def optimize_prompt_cache_prefix_alignment(
    prompt_segments: List[Dict[str, Any]],
    cache_block_size_tokens: int = 128
) -> str:
    """
    Orders prompt segments to place static immutable prefixes first and calculates KV cache alignment.
    """
    res = launcher.optimize_prompt_cache_prefix_alignment(prompt_segments, cache_block_size_tokens)
    return json.dumps(res, indent=2)

@mcp.tool()
def pack_context_token_budget_greedy(
    context_candidates: List[Dict[str, Any]],
    max_context_budget_tokens: int = 4096
) -> str:
    """
    Greedy packing of context elements into LLM context window based on value-to-cost ratio.
    """
    res = launcher.pack_context_token_budget_greedy(context_candidates, max_context_budget_tokens)
    return json.dumps(res, indent=2)

def main():
    print("Testing Context Launcher:")
    items = [{"id": "sys", "priority_score": 5.0}, {"id": "usr", "priority_score": 2.0}]
    print(allocate_attention_weights_softmax(items))

if __name__ == "__main__":
    main()
