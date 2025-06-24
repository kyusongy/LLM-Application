"""End-to-end demo of APIGen-MT Phase-1 + Phase-2.

1. Generates a validated task *blueprint* using the Phase-1 pipeline.
2. Feeds the blueprint to the Phase-2 trajectory collector, producing an
   interaction trajectory if the agent reaches the goal.

Adjust the LLM endpoint/model path and the tool schema/persona samples as
needed.  A successful run prints the conversation and dumps a JSON artefact to
`trajectory_example.json`.
"""

from pathlib import Path
import json

from common.llm import LLMConfig
from common.tool import ToolCalling
from blueprint.pipeline import generate_valid_blueprint
from trajectory.pipeline import TrajectoryCollector

# ---------------------------------------------------------------------------
# 0.  LLM SERVER CONFIG  ––  CHANGE THESE
# ---------------------------------------------------------------------------
CFG = LLMConfig(
    base_url="http://127.0.0.1:12345/v1",        # your OpenAI-compatible endpoint
    model="/data1/yaoys6/models/Qwen3-32B",                  # model name or path
    api_key="tokenabc123",                         # any string if server ignores auth
)

# ---------------------------------------------------------------------------
# 1.  SAMPLE TOOL SCHEMA & PLACEHOLDER DOMAIN INFO
# ---------------------------------------------------------------------------
API_SCHEMA = {
    "create_order": {
        "description": "Create a new order",
        "parameters": {
            "item": {"type": "string", "required": True},
            "quantity": {"type": "integer", "required": True},
        },
    },
    "track_order": {
        "description": "Track order status",
        "parameters": {"order_id": {"type": "string", "required": True}},
    },
}
PERSONAS = [
    "Casual shopper Alice",
    "Business client Bob",
]
DOMAIN_RULES = (
    "Orders with quantity > 5 are not allowed. No cancellations for digital goods."
)

# ---------------------------------------------------------------------------
# 2.  GENERATE & VALIDATE ONE BLUEPRINT  (Phase-1)
# ---------------------------------------------------------------------------
blueprint = generate_valid_blueprint(
    CFG,
    API_SCHEMA,
    PERSONAS,
    prompt_kwargs={
        "domain_rules": DOMAIN_RULES,
        "sampled_user_details": "- User ID: U123, Name: Alice, History: Bought books",
        "sampled_orders": "- Order ID: ORD789, Item: Headphones, Status: Shipped",
        "examples": "",  # optional few-shot example
    },
)

print("\n=== VALIDATED BLUEPRINT ===")
print(
    json.dumps(
        {
            "intent": blueprint.user_intent,
            "actions": [a.model_dump() for a in blueprint.actions],
            "outputs": blueprint.expected_outputs,
        },
        indent=2,
        ensure_ascii=False,
    )
)

# ---------------------------------------------------------------------------
# 3.  COLLECT TRAJECTORY  (Phase-2)
# ---------------------------------------------------------------------------
collector = TrajectoryCollector(CFG, CFG, tools_schema=API_SCHEMA, debug=True)
trajectory = collector.collect(blueprint)

if trajectory:
    print(f"\n✔ Collected trajectory with {len(trajectory.turns)} turns\n")
    for t in trajectory.turns:
        tag = "USER" if t.role == "user" else "ASSISTANT"
        print(f"[{tag}] {t.content}")

    artefact = {
        "blueprint": {
            "intent": blueprint.user_intent,
            "actions": [a.model_dump() for a in blueprint.actions],
            "outputs": blueprint.expected_outputs,
        },
        "trajectory": {
            "turns": [t.__dict__ for t in trajectory.turns],
            "tool_calls": [tc.model_dump() for tc in trajectory.tool_calls],
        },
    }
    Path("trajectory_example.json").write_text(
        json.dumps(artefact, indent=2, ensure_ascii=False)
    )
    print("\nSaved trajectory to trajectory_example.json")
else:
    print("✖ Failed to collect a valid trajectory") 