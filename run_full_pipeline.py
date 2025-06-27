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
# 0.  DOMAIN-SPECIFIC DUMMY TOOLS & DATA
# ---------------------------------------------------------------------------

from dummy_tools import TOOLS_SCHEMA  # noqa: E402 – after sys.path is set

# ---------------------------------------------------------------------------
# 1.  TOOL SCHEMA  (imported from dummy_tools)
# ---------------------------------------------------------------------------
API_SCHEMA = TOOLS_SCHEMA

# ---------------------------------------------------------------------------
# 2.  SIMPLE BUSINESS POLICY & PERSONAS
# ---------------------------------------------------------------------------

DOMAIN_RULES = "Delivered orders cannot be canceled. Exchanges only allowed within 30 days."

PERSONAS = [
    "Budget-conscious student Emma",  # prefers deals, concise questions
    "Detail-oriented engineer Yusuf",  # wants everything in one go
    "Busy parent Carla",               # cares about speed and convenience
]

# Dummy user & order snapshots – used as prompt context for the blueprint LLM
SAMPLED_USER_DETAILS = """
- User ID: user_YRos_19122, Name: Yusuf Rossi, ZIP: 19122, History: Bought peripherals & smart devices.
- User ID: user_EClar_27513, Name: Emma Clark, ZIP: 27513, History: Bought textbooks & earbuds.
"""

SAMPLED_ORDERS = """
- Order ID: #W2378156, Status: Delivered, Items: [keyboard-id 1151293680, thermostat-id 4983901480]
- Order ID: #X9934712, Status: Shipped,   Items: [earbuds-id 3311224488]
"""

# ---------------------------------------------------------------------------
# 3.  EXAMPLE BLUEPRINT (few-shot) — mirrors τ-bench structure
# ---------------------------------------------------------------------------

EXAMPLE_TASK = """
<thought>
The user has received order #W2378156 (keyboard & thermostat) and wants to
exchange both items for alternatives that better match their preferences.
To fulfil this we must:
  1) `find_user_id_by_name_zip` to get internal user ID;
  2) `get_order_details` to verify delivery status;
  3-4) `get_product_details` for each desired replacement item;
  5) `exchange_delivered_order_items` with correct mappings.
</thought>
<answer>
{
  "intent": "You are Yusuf Rossi in 19122. You received your order #W2378156 and wish to exchange the mechanical keyboard for a similar one but with clicky switches and the smart thermostat for one compatible with Google Home instead of Apple HomeKit. If there is no keyboard that is clicky, RGB backlight, full size, you'd go for no backlight. You are detail-oriented and want to make sure everything is addressed in one go.",
  "actions": [
    {
      "name": "find_user_id_by_name_zip",
      "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}
    },
    {
      "name": "get_order_details",
      "arguments": {"order_id": "#W2378156"}
    },
    {
      "name": "get_product_details",
      "arguments": {"product_id": "7706410293"}
    },
    {
      "name": "get_product_details",
      "arguments": {"product_id": "7747408585"}
    },
    {
      "name": "exchange_delivered_order_items",
      "arguments": {
        "order_id": "#W2378156",
        "item_ids": ["1151293680", "4983901480"],
        "new_item_ids": ["7706410293", "7747408585"],
        "payment_method_id": "credit_card_9513926"
      }
    }
  ],
  "outputs": [
    "Your exchange order is being processed."
  ]
}
</answer>
"""

# ---------------------------------------------------------------------------
# 4.  GENERATE BLUEPRINT (Phase-1)
# ---------------------------------------------------------------------------

blueprint = generate_valid_blueprint(
    CFG,
    API_SCHEMA,
    PERSONAS,
    prompt_kwargs={
        "domain_rules": DOMAIN_RULES,
        "sampled_user_details": SAMPLED_USER_DETAILS,
        "sampled_orders": SAMPLED_ORDERS,
        "examples": EXAMPLE_TASK,
        "task_rules": "",
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
# 5.  COLLECT TRAJECTORY  (Phase-2)
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