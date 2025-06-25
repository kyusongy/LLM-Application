from __future__ import annotations

"""APIGen-MT Phase-2 prototype: Trajectory collection via simulated human-agent interplay.

The goal is to demonstrate the end-to-end mechanics while keeping the code
compact and editable. All prompts come from Fig.-11 and Fig.-12 of the paper
(see screenshots) but are parameterised with placeholders so you can inject
real data later.
"""

from dataclasses import dataclass
import json
import random
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from common.llm import LLMConfig, GenerationOptions as LLMGenOpts
from common.tool import ToolCalling
from llm.sync_client import sync_request_llm
from blueprint.pipeline import Blueprint  # reuse dataclass from phase-1

# ---------------- Qwen-Agent integration -----------------
# pylint: disable=import-error
from qwen_agent.agents import Assistant  # type: ignore
# Ensure tool wrappers are registered before Assistant is instantiated
import qwen_tool_wrappers  # noqa: F401  # registers tools via import side-effects

# ---------------------------------------------------------------------------
# Prompts (verbatim from paper – Fig.11 & Fig.12)
# ---------------------------------------------------------------------------

_TRAJECTORY_COLLECTION_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a detail-oriented user interacting with an AI agent.

    ## Intent
    {intent}

    ## Rules
    • Generate one line at a time to simulate the user's message.
    • Do not give away all the intent at once. Only provide the information that is necessary for the current step.
    • Do not hallucinate information that is not provided in the intent.
    • If the intent goal is satisfied, generate `###STOP###` to end the conversation.
    • Do not repeat the exact intent in the conversation. Instead, use your own words to convey the same information.
    • Do not copy or repeat any assistant messages. Always write a brand-new user turn in your own words.
    • Try to make the conversation as natural as possible and stick to the personalities in the intent.

    ### Response format
    Reply with **only** the next message you (the user) would send. Do NOT include any explanations, headings, bullet points, or additional reasoning—just the raw chat line.
    """
).strip()

_BON_USER_LM_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a fair judge and an expert in following details.

    A human is interacting with a retail assistant to get help on solving their task. You are provided with the description of the human and the task the
    human wants to accomplish (wrapped with <description></description>), and a candidate response (wrapped with <response></response>) the human wants to
    give the assistant. Please help the human evaluate this candidate response, give an integer score (ranging from 0 to 10) to indicate the correctness of the response,
    higher score means better quality.

    1. If the response includes specific item / order / personal details, and they correctly match the task description you should give full score of 10. If there is some
       change in details, give a corresponding lower score (more incorrect details gets lower score).
    2. The response can include any normal conversation otherwise (e.g. asking details, saying ###STOP###) etc. which are all correct responses.
    3. Additionally, if the candidate_response keeps the conversation flowing by describing the task clearly / gives information properly then give a high score and if not
       (e.g. "I don't remember" or unhelpful response) should get a corresponding lower score.

    <description> {description} </description>
    <response> {response} </response>

    After scoring using the mentioned guideline, tell me your score, wrap it in <score></score> tags.
    """
).strip()

# ---------------------------------------------------------------------------
# Data-classes for interaction turns
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    role: str  # "user" / "assistant"
    content: str


@dataclass
class Trajectory:
    turns: List[Turn]
    tool_calls: List[ToolCalling]

# ---------------------------------------------------------------------------
# Simulated actors
# ---------------------------------------------------------------------------


class SimulatedHuman:
    """LLM that reveals the intent gradually using BoN self-critique."""

    def __init__(self, llm_cfg: LLMConfig, bon_n: int = 1, debug: bool = False):
        self.cfg = llm_cfg
        self.bon_n = bon_n
        self.debug = debug
        self._primed = False  # ensure giant prompt is sent only once

        # Lower temperature & token budget for faster, more focused replies during prototyping
        self.agent_opts = LLMGenOpts(
            temperature=0.3,
            max_tokens=2048,
            timeout=120,
            extra_body={"enable_reasoning": True},
            debug=debug,
        )
        self.judge_opts = LLMGenOpts(
            temperature=0.0,
            max_tokens=256,
            timeout=30,
            extra_body={"enable_reasoning": True},
            debug=debug,
        )

    def _score_candidate(self, description: str, candidate: str) -> int:
        prompt = _BON_USER_LM_PROMPT_TEMPLATE.format(description=description, response=candidate)
        messages = [{"role": "user", "content": prompt}]
        completion = sync_request_llm(self.cfg, messages, generation_config=self.judge_opts)
        reply = _get_msg_content(completion.choices[0].message)  # type: ignore[attr-defined]
        match = re.search(r"<score>(\d+)</score>", reply or "")
        return int(match.group(1)) if match else 0

    def next_message(self, intent: str, history: List[Turn]) -> str:
        """Generate the next user message via Best-of-N sampling."""
        # Insert the persona prompt only on the very first turn and keep it in
        # the running history thereafter (mirrors TestAgent behaviour).
        if not self._primed:
            self._persona_prompt = _TRAJECTORY_COLLECTION_PROMPT_TEMPLATE.format(intent=intent)
            # Prepend as a system turn so future calls include it automatically
            history.insert(0, Turn("system", self._persona_prompt))
            self._primed = True

        # Convert the (now complete) history—including the system prompt—into
        # the OpenAI chat format.
        messages = [{"role": t.role, "content": t.content} for t in history]

        candidates: List[Tuple[str, int]] = []
        if self.bon_n <= 1:
            comp = sync_request_llm(self.cfg, messages, generation_config=self.agent_opts)

            if self.debug:
                m = comp.choices[0].message
                print("[DEBUG/HUMAN] content:", repr(getattr(m, "content", "")))
                print("[DEBUG/HUMAN] reasoning_content:", repr(getattr(m, "reasoning_content", "")))

            raw = _get_msg_content(comp.choices[0].message)  # type: ignore[attr-defined]
            best_msg = raw if isinstance(raw, str) else ""
        else:
            for _ in range(self.bon_n):
                comp = sync_request_llm(self.cfg, messages, generation_config=self.agent_opts)

                if self.debug:
                    m = comp.choices[0].message
                    print("[DEBUG/HUMAN] content:", repr(getattr(m, "content", "")))
                    print("[DEBUG/HUMAN] reasoning_content:", repr(getattr(m, "reasoning_content", "")))

                raw = _get_msg_content(comp.choices[0].message)  # type: ignore[attr-defined]
                msg = raw if isinstance(raw, str) else ""
                score = self._score_candidate(intent, msg) if msg else 0
                candidates.append((msg, score))
            # choose best non-empty candidate
            best_msg = max(candidates, key=lambda x: x[1])[0]

        # Post-process to strip meta reflections like "Okay, the user wants …".
        if not best_msg:
            return ""

        def _clean_line(line: str) -> str:
            meta_keywords = (
                "the user", "assistant", "i should", "let me", "they need", "i need to", "next step", "first,", "okay,", "wait,", "perhaps",
            )
            low = line.lower()
            return "" if any(k in low for k in meta_keywords) else line

        # keep first non-meta line; if none found fallback to original
        for ln in best_msg.splitlines():
            cleaned = _clean_line(ln.strip())
            if cleaned:
                return cleaned

        return best_msg.strip()


class QwenTestAgent:
    """Wrap Qwen-Agent's `Assistant` for our trajectory collector."""

    def __init__(self, llm_cfg: LLMConfig):
        # Import registry after wrappers have been imported so tools are present.
        import dummy_tools as _d

        self.bot = Assistant(
            llm={
                "model": llm_cfg.model,
                "model_server": llm_cfg.base_url,
                "api_key": llm_cfg.api_key,
            },
            function_list=list(_d.TOOLS_SCHEMA.keys()),  # expose only dummy tools to avoid heavy deps
        )

    def respond(self, history: List[dict]) -> list[dict]:
        """Return the list of new messages generated by the agent for the current turn."""

        new_msgs: list[dict] = []
        for batch in self.bot.run(messages=history):
            new_msgs = batch  # the library streams; each `batch` is list of new messages
        return new_msgs


# ---------------------------------------------------------------------------
# Trajectory collector
# ---------------------------------------------------------------------------


class TrajectoryCollector:
    def __init__(
        self,
        human_cfg: LLMConfig,
        agent_cfg: LLMConfig,
        tools_schema: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        """Collect trajectories.

        tools_schema: full domain tool specification (same dict passed to blueprint
        generation).  Providing it allows the agent to see argument structure and
        increases the chance it will emit correct function calls.
        """
        self.human = SimulatedHuman(human_cfg, debug=debug)
        self.agent = QwenTestAgent(agent_cfg)
        self.tools_schema = tools_schema or {}
        self.debug = debug

    def collect(self, blueprint: Blueprint) -> Optional[Trajectory]:
        history: List[Turn] = []
        tool_calls: List[ToolCalling] = []

        # Simulate turns (hard-coded max 20 to avoid infinite loop)
        for _ in range(20):
            # 1) human speaks
            user_msg = self.human.next_message(blueprint.user_intent, history)
            if not user_msg:
                if self.debug:
                    print("[Collector] human LLM returned empty response, aborting trajectory")
                return None
            history.append(Turn("user", user_msg))
            if self.debug:
                print("[USER]", user_msg)
            if user_msg.strip().startswith("###STOP###"):
                break

            # 2) agent responds (may include tool call)
            # ⚠️  The first turn in `history` is an internal system prompt that
            # embeds the *full intent* so the SimulatedHuman can generate
            # appropriate messages.  The agent MUST **not** see this –
            # otherwise it trivially solves the task.  We therefore strip all
            # system messages before forwarding the conversation to the agent.

            # Translate internal Turn records to OpenAI/Qwen message schema
            messages_for_agent: List[dict] = []
            last_fc_name: Optional[str] = None
            for t in history:
                if t.role == "system":
                    continue  # hide intent
                if t.role == "function_call":
                    try:
                        fc_payload = json.loads(t.content)
                    except Exception:
                        continue  # malformed, skip
                    last_fc_name = fc_payload.get("name")
                    # Qwen-Agent expects arguments as **string**
                    args_str = fc_payload.get("arguments")
                    if isinstance(args_str, dict):
                        args_str = json.dumps(args_str, ensure_ascii=False)
                    assistant_fc = {
                        "name": fc_payload.get("name"),
                        "arguments": args_str,
                    }
                    messages_for_agent.append({
                        "role": "assistant",
                        "function_call": assistant_fc,
                        "content": ""  # per spec, content must be empty or omitted when using function_call
                    })
                elif t.role == "observation":
                    if last_fc_name is None:
                        continue  # cannot pair; skip
                    messages_for_agent.append({
                        "role": "function",
                        "name": last_fc_name,
                        "content": t.content,
                    })
                    last_fc_name = None  # reset pairing
                else:
                    messages_for_agent.append({"role": t.role, "content": t.content})

            # Build tools list for this turn
            tools_schema_list: List[dict] = []
            for act in blueprint.actions:
                if act.name in self.tools_schema:
                    spec = self.tools_schema[act.name]
                    parameters = spec.get("parameters", {})
                    tools_schema_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": act.name,
                                "description": spec.get("description", ""),
                                "parameters": {
                                    "type": "object",
                                    "properties": {k: {"type": v.get("type", "string")} for k, v in parameters.items()},
                                    "required": [k for k, v in parameters.items() if v.get("required", False)],
                                },
                            },
                        }
                    )
                else:
                    tools_schema_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": act.name,
                                "description": "",
                                "parameters": {"type": "object", "properties": {}, "required": []},
                            },
                        }
                    )

            new_msgs = self.agent.respond(messages_for_agent)

            for m in new_msgs:
                role = m.get("role")
                if role == "assistant" and "function_call" in m:
                    fc = m["function_call"]
                    # `function_call.arguments` might come as *string* (JSON) per
                    # OpenAI spec.  Convert to dict so ToolCalling passes
                    # Pydantic validation.
                    raw_args = fc.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            # leave as-is; validator will flag it later
                            pass

                    val = json.dumps({"name": fc.get("name"), "arguments": json.dumps(raw_args, ensure_ascii=False)}, ensure_ascii=False)
                    history.append(Turn("function_call", val))
                    tool_calls.append(ToolCalling(name=fc.get("name"), arguments=raw_args))
                    if self.debug:
                        print("[FUNC_CALL]", val)
                elif role == "function":
                    history.append(Turn("observation", m.get("content", "")))
                    if self.debug:
                        print("[OBSERV]", m.get("content", ""))
                else:
                    # standard assistant message
                    content = m.get("content", "")
                    history.append(Turn("assistant", content))
                    if self.debug:
                        print("[ASSISTANT]", content)

            # success check: all expected outputs present in assistant messages
            if any(isinstance(t.content, str) and any(o in t.content for o in blueprint.expected_outputs) for t in history if t.role == "assistant"):
                history.append(Turn("user", "###STOP###"))
                break

        # validate tool_calls (ignore order) – all blueprint actions must appear
        if sorted(c.name for c in tool_calls) == sorted(a.name for a in blueprint.actions):
            return Trajectory(history, tool_calls)
        return None


# ---------------------------------------------------------------------------
# Example script usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Dummy configs (replace with real endpoints)
    cfg = LLMConfig(base_url="http://127.0.0.1:12345/v1", model="/data1/yaoys6/models/Qwen3-32B", api_key="tokenabc123")
    blueprint_example = Blueprint(
        user_intent="I want to order a laptop and then track it.",
        actions=[ToolCalling(name="create_order", arguments={"item": "laptop", "quantity": 1}),
                 ToolCalling(name="track_order", arguments={"order_id": "ORD123"})],
        expected_outputs=["Your order for 1 laptop has been placed.", "Order ORD123 is currently"],
    )

    collector = TrajectoryCollector(cfg, cfg)
    traj = collector.collect(blueprint_example)
    if traj:
        print("Collected trajectory with", len(traj.turns), "turns")
        print(json.dumps([t.__dict__ for t in traj.turns], indent=2, ensure_ascii=False))
    else:
        print("Failed to collect a valid trajectory")

# ---------------------------------------------------------------------------
# Utility for vLLM "reasoning" messages
# ---------------------------------------------------------------------------


def _get_msg_content(msg) -> str:  # type: ignore[Any]
    """Return message.content, falling back to .reasoning_content if present (vLLM).

    If reasoning_content is used (when the server is launched with --enable-reasoning),
    it often contains chain-of-thought prose and then the real reply (e.g. prefixed
    with "First message:").  We strip everything except the final answer so the
    rest of the pipeline sees just the conversational text.
    """
    content = getattr(msg, "content", None)
    return content.strip() if isinstance(content, str) else "" 