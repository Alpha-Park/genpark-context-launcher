# -*- coding: utf-8 -*-
"""
GenPark Context Launcher Client
Pure Python Standard Library (Zero External Dependencies)
Implements dynamic attention weight softmax allocation, prefix cache boundary block alignment,
and greedy token budget knapsack context packaging.
"""

import math
from typing import Dict, Any, List, Optional

class GenParkContextLauncher:
    """
    LLM prompt context window optimization, attention allocation, and KV cache alignment engine.
    """

    def __init__(self):
        pass

    def allocate_attention_weights_softmax(
        self,
        context_items: List[Dict[str, Any]],
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Computes normalized attention allocation weights across context layers using Softmax:
          w_i = exp( priority_i / T ) / sum( exp( priority_j / T ) )
        """
        if not context_items:
            return {"weights": [], "entropy": 0.0}

        temp = max(0.01, float(temperature))
        scores = [float(item.get("priority_score", 1.0)) for item in context_items]
        
        # Numerical stability shift
        max_s = max(scores)
        exp_vals = [math.exp((s - max_s) / temp) for s in scores]
        sum_exp = sum(exp_vals)
        weights = [round(v / sum_exp, 4) for v in exp_vals]

        # Shannon entropy of weights
        entropy = -sum(w * math.log2(w) for w in weights if w > 0)

        results = []
        for i, item in enumerate(context_items):
            results.append({
                "item_id": item.get("id", f"item_{i}"),
                "priority_score": scores[i],
                "attention_weight": weights[i]
            })

        return {
            "temperature": temp,
            "attention_entropy": round(entropy, 3),
            "allocated_weights": results
        }

    def optimize_prompt_cache_prefix_alignment(
        self,
        prompt_segments: List[Dict[str, Any]],
        cache_block_size_tokens: int = 128
    ) -> Dict[str, Any]:
        """
        Orders prompt segments to place static immutable prefixes first and calculates
        boundary padding to align with KV cache block boundaries (e.g. 128 tokens for Anthropic/DeepSeek).
        """
        # Sort segments: static / immutable first, dynamic session state last
        sorted_segments = sorted(
            prompt_segments,
            key=lambda s: 0 if s.get("is_static", False) else 1
        )

        aligned_blocks = []
        cumulative_tokens = 0

        for seg in sorted_segments:
            seg_tokens = max(1, int(seg.get("token_count", 50)))
            cumulative_tokens += seg_tokens
            
            # Boundary calculation
            remainder = cumulative_tokens % cache_block_size_tokens
            tokens_to_next_block = (cache_block_size_tokens - remainder) % cache_block_size_tokens

            aligned_blocks.append({
                "segment_id": seg.get("id", "seg"),
                "is_static": seg.get("is_static", False),
                "token_count": seg_tokens,
                "cumulative_tokens": cumulative_tokens,
                "aligned_to_cache_block": remainder == 0,
                "tokens_to_pad_for_perfect_alignment": tokens_to_next_block
            })

        static_tokens = sum(s["token_count"] for s in aligned_blocks if s["is_static"])
        cacheable_ratio = round((static_tokens / max(1, cumulative_tokens)) * 100.0, 1)

        return {
            "total_tokens": cumulative_tokens,
            "cache_block_size": cache_block_size_tokens,
            "static_cacheable_tokens": static_tokens,
            "cacheable_prefix_ratio_pct": cacheable_ratio,
            "alignment_plan": aligned_blocks
        }

    def pack_context_token_budget_greedy(
        self,
        context_candidates: List[Dict[str, Any]],
        max_context_budget_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Greedy packing of context elements into LLM context window based on value-to-cost (priority / tokens) ratio.
        """
        # Calculate ratio
        items = []
        for c in context_candidates:
            tokens = max(1, int(c.get("token_count", 100)))
            val = float(c.get("value_score", 1.0))
            ratio = val / tokens
            items.append({
                "id": c.get("id", "ctx"),
                "token_count": tokens,
                "value_score": val,
                "efficiency_ratio": round(ratio, 4),
                "content": c.get("content", "")
            })

        # Sort descending by efficiency ratio
        sorted_items = sorted(items, key=lambda x: x["efficiency_ratio"], reverse=True)

        packed = []
        consumed_tokens = 0
        total_value = 0.0

        for it in sorted_items:
            if consumed_tokens + it["token_count"] <= max_context_budget_tokens:
                packed.append(it)
                consumed_tokens += it["token_count"]
                total_value += it["value_score"]

        return {
            "budget_tokens": max_context_budget_tokens,
            "consumed_tokens": consumed_tokens,
            "remaining_token_headroom": max_context_budget_tokens - consumed_tokens,
            "utilization_pct": round((consumed_tokens / max_context_budget_tokens) * 100.0, 1),
            "total_value_packed": round(total_value, 2),
            "packed_items_count": len(packed),
            "packed_items": packed
        }
