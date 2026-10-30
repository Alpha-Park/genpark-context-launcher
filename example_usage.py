# -*- coding: utf-8 -*-
from client import GenParkContextLauncher

launcher = GenParkContextLauncher()

# 1. Softmax Attention Weighting
layers = [
    {"id": "system_prompt_rules", "priority_score": 8.0},
    {"id": "few_shot_examples", "priority_score": 5.0},
    {"id": "user_query", "priority_score": 9.0},
    {"id": "retrieved_rag_context", "priority_score": 6.5}
]
weights = launcher.allocate_attention_weights_softmax(layers, temperature=1.5)
print(f"Attention Softmax: Entropy={weights['attention_entropy']}")
for w in weights['allocated_weights']:
    print(f"  {w['item_id']}: Weight={w['attention_weight']} (Priority={w['priority_score']})")

# 2. KV Cache Prefix Alignment
segments = [
    {"id": "session_chat_history", "token_count": 210, "is_static": False},
    {"id": "system_instructions", "token_count": 256, "is_static": True},
    {"id": "tool_definitions_schema", "token_count": 384, "is_static": True}
]
cache = launcher.optimize_prompt_cache_prefix_alignment(segments, cache_block_size_tokens=128)
print()
print(f"KV Cache Alignment: Cacheable Ratio={cache['cacheable_prefix_ratio_pct']}% ({cache['static_cacheable_tokens']}/{cache['total_tokens']} tokens)")
for b in cache['alignment_plan']:
    print(f"  {b['segment_id']} (Static={b['is_static']}): CumTokens={b['cumulative_tokens']}, PadToAlign={b['tokens_to_pad_for_perfect_alignment']}")

# 3. Greedy Context Token Packing
rag_docs = [
    {"id": "doc_api_spec", "token_count": 450, "value_score": 95.0},
    {"id": "doc_faq", "token_count": 200, "value_score": 30.0},
    {"id": "doc_auth_guide", "token_count": 300, "value_score": 85.0},
    {"id": "doc_changelog", "token_count": 600, "value_score": 15.0}
]
packed = launcher.pack_context_token_budget_greedy(rag_docs, max_context_budget_tokens=800)
print()
print(f"Token Budget Packing: {packed['consumed_tokens']}/{packed['budget_tokens']} tokens ({packed['utilization_pct']}%)")
for p in packed['packed_items']:
    print(f"  + {p['id']}: Tokens={p['token_count']}, Value={p['value_score']} (Eff={p['efficiency_ratio']})")
