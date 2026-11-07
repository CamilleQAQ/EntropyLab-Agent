"""LangGraph research loop: plan -> execute -> review -> (plan | finalize)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from . import grounding
from .custom import catalogue as custom_catalogue
from .ledger import Ledger
from .llm import chat
from .reviewer import review_experiment
from .tools import TOOL_SPECS, LabContext, tool_catalogue


class Action(BaseModel):
    action: Literal[
        "profile_dataset",
        "run_experiment",
        "analyze_errors",
        "search_notes",
        "run_custom_experiment",
        "finish",
        "ask_human",
    ]
    args: dict = Field(default_factory=dict)
    hypothesis: str = ""
    claim: str = ""
    reason: str = ""


class LabState(TypedDict, total=False):
    goal: str
    max_rounds: int
    rounds: int
    budget: float
    spent: float
    profile: dict | None
    history: list[dict]
    hypothesis: str
    claim: str
    action: dict
    result: dict | None
    review: dict
    baseline: dict | None
    status: str
    report: str
    hypothesis_problems: list[str]


RISKY_ACTIONS = {"run_custom_experiment", "finish"}

# Ablation switches (read once at import): used by scripts/run_ablation.py.
DISABLE_GROUNDING = os.environ.get("LAB_DISABLE_GROUNDING") == "1"
DISABLE_REVIEWER = os.environ.get("LAB_DISABLE_REVIEWER") == "1"


SYSTEM_PROMPT = """You are the Research Director of an autonomous bearing-diagnosis lab.
You run controlled experiments on a vibration dataset and must earn every claim.

Rules:
- Reply with ONE JSON object and nothing else.
- Shape: {"action": <tool or finish/ask_human>, "args": {...}, "hypothesis": "...", "claim": "...", "reason": "..."}
- Use search_notes before proposing a hypothesis about method behaviour.
- Use run_experiment with seeds >= 3, otherwise the reviewer will reject it.
- A "claim" is only allowed when the difference exceeds twice the split variation.
- When evidence is exhausted or the budget is nearly spent, call finish.
- Never invent numbers: every number you mention must come from a tool result.
- Never repeat an experiment that already ran: pick a feature/classifier pair that
  has not been tried, otherwise the call is rejected as REPEATED_ACTION.
- "hypothesis" must be at most two sentences and follow this shape:
  "观察到 <工具返回的具体数字>，因此预计 <可检验的变化>，用 <检验方式> 验证。"
  A hypothesis that cites a number you did not see in the history is rejected and
  you will be asked to rewrite it.
  Example: "观察到 INNER_RING recall 0.42、OR->IR 混淆 12 例，因此预计加入包络特征后
  INNER_RING recall 提升到 0.6 以上，用 5 个划分的均值对比验证。"

Tools:
"""


def _compact(history: list[dict]) -> str:
    lines = []
    for item in history[-8:]:
        if item.get("error"):
            lines.append(f"- {item['tool']} FAILED: {item['error'][:120]}")
        elif item.get("tool") == "run_experiment" and "macro_f1_mean" in item:
            lines.append(
                f"- run_experiment {item['feature']}+{item['classifier']} seeds={item['seeds']} "
                f"macro_f1={item['macro_f1_mean']:.4f}+-{item['macro_f1_std']:.4f}"
            )
        elif item.get("tool") == "analyze_errors":
            pairs = "; ".join(
                f"{p['truth']}->{p['predicted']} x{p['count']}" for p in item["top_confusions"]
            )
            lines.append(f"- analyze_errors {item['feature']}+{item['classifier']}: {pairs}")
        elif item.get("tool") == "search_notes":
            lines.append(f"- search_notes '{item['query']}': {len(item['hits'])} hits")
        elif item.get("tool") == "profile_dataset":
            lines.append(
                f"- profile: records={item['records']} classes={len(item['classes'])} "
                f"groups={item['groups']}"
            )
        elif item.get("tool") == "review":
            lines.append(f"- reviewer: {item['verdict']} {[f['code'] for f in item['findings']]}")
    return "\n".join(lines) or "(no experiments yet)"


def _parse_action(text: str) -> Action:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    return Action.model_validate(json.loads(cleaned))


def build_graph(ctx: LabContext, llm=chat, checkpointer=None):
    ledger = Ledger(ctx.workspace / "ledger.sqlite")

    def plan(state: LabState) -> LabState:
        if state.get("rounds", 0) >= state.get("max_rounds", 6):
            return {"status": "finish", "report": ""}
        if state.get("spent", 0.0) >= state.get("budget", ctx.budget_seconds):
            return {"status": "finish", "report": ""}

        profile = state.get("profile")
        history = state.get("history", [])
        ran_experiment = any(item.get("tool") == "run_experiment" for item in history)
        hint = ""
        if not ran_experiment:
            hint = (
                "\nIMPORTANT: no experiment has been run yet. The next action MUST be "
                "run_experiment with seeds >= 3 (pick a feature and a classifier from the "
                "tool schema). Do not call search_notes or profile_dataset again."
            )
        prompt = (
            f"Goal: {state['goal']}\n"
            f"Profile: {json.dumps(profile, ensure_ascii=False) if profile else 'not inspected yet'}\n"
            f"History:\n{_compact(state.get('history', []))}\n"
            f"Round {state.get('rounds', 0) + 1}/{state.get('max_rounds', 6)}; "
            f"spent {state.get('spent', 0.0):.0f}s of {state.get('budget', ctx.budget_seconds):.0f}s.\n"
            "Choose the single next action."
            + hint
            + "\nNever repeat the same tool with the same arguments twice."
        )
        registered = custom_catalogue(ctx.workspace)
        extra = ""
        if registered:
            names = ", ".join(item["name"] for item in registered)
            extra = (
                "\nRegistered custom feature extractors (usable with "
                f"run_custom_experiment): {names}"
            )
        raw = llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT + tool_catalogue() + extra},
                {"role": "user", "content": prompt},
            ]
        )
        action = _parse_action(raw)
        hypothesis_problems = (
            []
            if DISABLE_GROUNDING
            else (grounding.problems(action.hypothesis, history) if action.hypothesis else [])
        )
        if hypothesis_problems:
            repair = (
                "Your hypothesis was rejected by the grounding check: "
                f"{hypothesis_problems}. Rewrite the SAME action, keeping the same "
                "tool and arguments, with a hypothesis that cites only numbers that "
                "appear in the history below.\n"
                + _compact(history)
            )
            try:
                repaired = _parse_action(
                    llm(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT + tool_catalogue()},
                            {"role": "user", "content": repair},
                        ]
                    )
                )
                if not grounding.problems(repaired.hypothesis, history):
                    repaired.action = action.action
                    repaired.args = action.args
                    action = repaired
                    hypothesis_problems = []
            except Exception:
                pass
        if action.hypothesis:
            evidence = action.reason
            if hypothesis_problems:
                evidence = f"{evidence} | 未通过校验: {hypothesis_problems}"
            ledger.record_hypothesis(state.get("rounds", 0) + 1, action.hypothesis, evidence)
        return {
            "action": action.model_dump(),
            "hypothesis": action.hypothesis,
            "claim": action.claim,
            "rounds": state.get("rounds", 0) + 1,
            "status": "planned" if action.action not in {"finish", "ask_human"} else action.action,
            "hypothesis_problems": hypothesis_problems,
        }

    def guard(state: LabState) -> LabState:
        """Human-in-the-loop gate: pause before a risky or concluding action."""
        action = state.get("action", {})
        name = action.get("action")
        if checkpointer is not None and name in RISKY_ACTIONS:
            answer = interrupt(
                {
                    "question": f"Approve `{name}` with args {action.get('args')}?",
                    "action": name,
                    "args": action.get("args"),
                    "hypothesis": state.get("hypothesis", ""),
                }
            )
            if not answer:
                return {
                    "status": "rejected",
                    "action": {
                        "action": "finish",
                        "args": {},
                        "hypothesis": action.get("hypothesis", ""),
                        "claim": "",
                        "reason": "human rejected the action",
                    },
                }
        return {}

    def execute(state: LabState) -> LabState:
        action = state.get("action", {})
        name = action.get("action")
        history = list(state.get("history", []))
        if name in {"finish", "ask_human"} or name not in TOOL_SPECS:
            return {"history": history}
        spec = TOOL_SPECS[name]
        signature = {"tool": name, "args": action.get("args") or {}}
        if any(
            item.get("_signature") == signature for item in history if item.get("_signature")
        ):
            result = {
                "tool": name,
                "error": "REPEATED_ACTION: this exact call already ran; choose a different action",
            }
        else:
            try:
                args = spec["args"].model_validate(action.get("args") or {})
                result = spec["fn"](ctx, args)
            except Exception as error:  # a bad tool call must not kill the run
                result = {"tool": name, "error": f"{type(error).__name__}: {error}"}
            result["_signature"] = signature
        ledger.record_experiment(state.get("rounds", 0), name, result)
        history.append(result)
        if name == "profile_dataset":
            return {"profile": result, "history": history, "spent": ctx.spent_seconds, "result": result}
        return {"history": history, "spent": ctx.spent_seconds, "result": result}

    def review(state: LabState) -> LabState:
        result = state.get("result") or {}
        baseline = state.get("baseline")
        verdict = (
            {"verdict": "accept", "findings": []}
            if DISABLE_REVIEWER
            else review_experiment(result, baseline, state.get("claim", ""))
        )
        history = list(state.get("history", []))
        history.append({"tool": "review", **verdict})
        ledger.record_review(state.get("rounds", 0), verdict["verdict"], verdict["findings"])
        update: LabState = {"review": verdict, "history": history}
        if result.get("tool") == "run_experiment" and result.get("seeds", 0) >= 3 and baseline is None:
            update["baseline"] = result
        rounds = state.get("rounds", 0)
        spent = state.get("spent", 0.0)
        if rounds >= state.get("max_rounds", 6) or spent >= state.get("budget", ctx.budget_seconds):
            update["status"] = "finish"
        return update

    def finalize(state: LabState) -> LabState:
        lines = [
            "# 研究日志",
            "",
            f"目标：{state.get('goal', '')}",
            "",
            "## 数据画像",
            "",
            f"```json\n{json.dumps(state.get('profile'), ensure_ascii=False, indent=2)}\n```",
            "",
            "## 实验与审查",
            "",
        ]
        for item in state.get("history", []):
            if item.get("error"):
                lines.append(f"- {item['tool']} 失败：{item['error'][:120]}")
            elif item.get("tool") == "run_experiment" and "macro_f1_mean" in item:
                lines.append(
                    f"- `{item['feature']}+{item['classifier']}`：macro-F1 "
                    f"{item['macro_f1_mean']:.4f} ± {item['macro_f1_std']:.4f}（{item['seeds']} 划分）"
                )
            elif item.get("tool") == "review":
                codes = [finding["code"] for finding in item["findings"]]
                lines.append(f"- 审查：{item['verdict']} {codes}")
        lines += ["", "## 假设", ""]
        for row in ledger.hypotheses():
            flag = " ⚠" if "未通过校验" in (row["evidence"] or "") else ""
            lines.append(
                f"- 第 {row['round']} 轮：{row['text']}（依据：{row['evidence']}）{flag}"
            )
        lines += [
            "",
            "## 局限",
            "",
            "- 只用了验证集，测试集未打开；",
            "- 结论仅覆盖本次运行的数据划分。",
        ]
        return {"report": "\n".join(lines), "status": "done"}

    def after_plan(state: LabState) -> str:
        return "guard"

    def after_guard(state: LabState) -> str:
        if state.get("status") == "rejected":
            return "finalize"
        action = (state.get("action") or {}).get("action")
        if action in {"finish", "ask_human"}:
            return "finalize"
        return "execute"

    def after_review(state: LabState) -> str:
        return "finalize" if state.get("status") == "finish" else "plan"

    graph = StateGraph(LabState)
    graph.add_node("plan", plan)
    graph.add_node("guard", guard)
    graph.add_node("execute", execute)
    graph.add_node("review", review)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", after_plan, {"guard": "guard", "finalize": "finalize"})
    graph.add_conditional_edges("guard", after_guard, {"execute": "execute", "finalize": "finalize"})
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", after_review, {"plan": "plan", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer), ledger


def _default_ask(payload: dict) -> bool:
    print(f"\n[human-in-the-loop] {payload.get('question')}")
    reply = input("approve? [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def run_lab(
    ctx: LabContext,
    *,
    goal: str,
    max_rounds: int = 6,
    llm=chat,
    interactive: bool = False,
    ask=None,
    thread_id: str = "lab",
) -> dict:
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver() if interactive else None
    graph, ledger = build_graph(ctx, llm=llm, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}} if interactive else None
    initial: LabState = {
        "goal": goal,
        "max_rounds": max_rounds,
        "rounds": 0,
        "budget": ctx.budget_seconds,
        "spent": 0.0,
        "profile": None,
        "history": [],
        "status": "start",
        "baseline": None,
    }
    if interactive:
        final = graph.invoke(initial, config)
        while "__interrupt__" in final:
            payload = final["__interrupt__"][0].value
            answer = (ask or _default_ask)(payload)
            final = graph.invoke(Command(resume=answer), config)
    else:
        final = graph.invoke(initial)
    report_path = ctx.workspace / "report.md"
    report_path.write_text(final.get("report", ""), encoding="utf-8")
    return {
        "status": final.get("status"),
        "rounds": final.get("rounds"),
        "spent_seconds": round(float(final.get("spent", 0.0)), 2),
        "report": str(report_path),
        "hypotheses": ledger.hypotheses(),
        "experiments": ledger.experiments(),
        "reviews": ledger.reviews(),
    }
