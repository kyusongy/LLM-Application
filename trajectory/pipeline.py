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
from openai import BadRequestError

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

    def __init__(self, llm_cfg: LLMConfig, bon_n: int = 2):
        self.cfg = llm_cfg
        self.bon_n = bon_n
        self._primed = False  # ensure giant prompt is sent only once
        # Lower temperature & token budget for faster, more focused replies during prototyping
        self.agent_opts = LLMGenOpts(
            temperature=0.3,
            max_tokens=64,
            timeout=60,
            extra_body={"enable_reasoning": False},
        )
        self.judge_opts = LLMGenOpts(
            temperature=0.0,
            max_tokens=32,
            timeout=30,
            extra_body={"enable_reasoning": False},
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
        if not self._primed:
            # First turn: send the giant prompt as a *user* message, no prior history
            messages = [{"role": "user", "content": _TRAJECTORY_COLLECTION_PROMPT_TEMPLATE.format(intent=intent)}]
            self._primed = True
        else:
            messages = [{"role": t.role, "content": t.content} for t in history]

        candidates: List[Tuple[str, int]] = []
        for _ in range(self.bon_n):
            comp = sync_request_llm(self.cfg, messages, generation_config=self.agent_opts)
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


class TestAgent:
    """LLM in function-calling mode that tries to satisfy the task."""

    def __init__(self, llm_cfg: LLMConfig):
        self.cfg = llm_cfg
        self.opts = LLMGenOpts(
            temperature=0.3,
            max_tokens=512,
            timeout=240,
            extra_body={"enable_reasoning": False},
        )
        self._primed = False

        self._assistant_prompt = (
            "You are an AI assistant that MUST satisfy the user's request **only** by calling one of the provided tools when required. "
            "Use the exact function name as listed; do NOT invent new names. "
            "Respond either with a JSON function call or, if no tool is needed, with a brief natural-language answer."
        )

    def respond(
        self,
        history: List[Dict],
        tools: Optional[List[dict]] = None,
    ) -> Tuple[str, Optional[ToolCalling]]:
        """Query the LLM and optionally enable OpenAI function-calling with the given tools."""

        # Prime once with assistant prompt as a system message
        if not self._primed:
            history = [{"role": "system", "content": self._assistant_prompt}] + history
            self._primed = True

        try:
            comp = sync_request_llm(self.cfg, history, tools=tools, generation_config=self.opts)
        except BadRequestError as e:
            if "auto" in str(e) and "tool choice" in str(e):
                # Retry with tool_choice disabled for servers that don't support auto choice
                opts_no_tool = self.opts.copy()
                opts_no_tool.extra_body = {**(opts_no_tool.extra_body or {}), "tool_choice": "none", "enable_reasoning": False}
                comp = sync_request_llm(self.cfg, history, tools=tools, generation_config=opts_no_tool)
            else:
                raise
        msg = comp.choices[0].message
        content = _get_msg_content(msg)

        tool_call: Optional[ToolCalling] = None
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                tool_call = ToolCalling(name=tc.function.name, arguments=json.loads(tc.function.arguments))
            except Exception:
                # fallback – malformed JSON or missing args
                tool_call = ToolCalling(name=tc.function.name, arguments={})

        return content or "", tool_call


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
        self.human = SimulatedHuman(human_cfg)
        self.agent = TestAgent(agent_cfg)
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
            messages_for_agent = [{"role": t.role, "content": t.content} for t in history]

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

            agent_text, call = self.agent.respond(messages_for_agent, tools=tools_schema_list)
            history.append(Turn("assistant", agent_text))
            if self.debug:
                print("[ASSISTANT]", agent_text)
            if call:
                tool_calls.append(call)
                if self.debug:
                    print("   ↳ tool_call", call.model_dump())

            # simple success check: if agent text contains any expected output string we break
            if any(out in agent_text for out in blueprint.expected_outputs):
                history.append(Turn("user", "###STOP###"))
                break

            # If we've observed all required tool calls we can finish early
            if sorted(c.name for c in tool_calls) == sorted(a.name for a in blueprint.actions):
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
    # 1) normal OpenAI behaviour
    content = getattr(msg, "content", None)
    if content and content.strip():
        return content.strip()

    # 2) vLLM reasoning mode – clean it up
    raw = getattr(msg, "reasoning_content", "").strip()
    if not raw:
        return ""

    import re, textwrap

    # Split into paragraphs delimited by blank lines
    paragraphs = re.split(r"\n\s*\n", raw)

    # Look from the end for a marker like "First message:" / "Next message:" / "Message:"
    marker_regex = re.compile(r"(?:first|next)?\s*message:\s*(.*)$", re.I | re.S)
    for para in reversed(paragraphs):
        m = marker_regex.search(para.strip())
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return textwrap.dedent(candidate).strip()

    # Otherwise just return the last non-empty paragraph
    for para in reversed(paragraphs):
        if para.strip():
            return para.strip()

    # Fallback: raw text
    return raw 