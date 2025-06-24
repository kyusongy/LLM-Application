#!/usr/bin/env python
"""Quick diagnostic script: send the Phase-2 first-turn prompt to the local Qwen server
and print the raw reply.

Run with:
    python debug_first_turn.py
Adjust LLM endpoint / api_key / model path as needed.
"""

from rich import print as rprint

from common.llm import LLMConfig, GenerationOptions as LLMGenOpts
from llm.sync_client import sync_request_llm
from trajectory.pipeline import _TRAJECTORY_COLLECTION_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Config – edit these if your server address or model differs
# ---------------------------------------------------------------------------
CFG = LLMConfig(
    api_key="tokenabc123",
    base_url="http://127.0.0.1:12345/v1",
    model="/data1/yaoys6/models/Qwen3-32B",
)

# Example user intent you want to test
INTENT = "I want to order a laptop and then track it."

prompt = _TRAJECTORY_COLLECTION_PROMPT_TEMPLATE.format(intent=INTENT)

messages = [
    {
        "role": "user",
        "content": prompt,
    }
]

opts = LLMGenOpts(temperature=0.3, max_tokens=128, timeout=60, extra_body={"enable_reasoning": False})

completion = sync_request_llm(CFG, messages, generation_config=opts)

# Print full message object so we can see content and any reasoning_content leakage
msg = completion.choices[0].message
rprint(msg) 