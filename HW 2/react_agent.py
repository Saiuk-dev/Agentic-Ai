"""
SmartTask — ReAct agent using local Ollama + Qwen2.5:3b.

Setup:
    ollama list
    pip install requests
    python react_agent.py

Offline controller demonstration (does not call Ollama):
    python react_agent.py --demo

If your Ollama model tag differs:
    python react_agent.py --model qwen2.5:3b

Ollama's default local API is http://localhost:11434.
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    import requests
except ImportError:
    requests = None


@dataclass
class ToolResult:
    success: bool
    output: str


@dataclass
class AgentState:
    goal: str
    plan: List[str] = field(default_factory=list)
    completed: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    replanned: bool = False
    steps: int = 0


class Tools:
    """Tools that the LLM can choose and execute."""

    def __init__(self):
        self.primary_resource_attempts = 0

    def fetch_primary_resource(self, topic: str) -> ToolResult:
        self.primary_resource_attempts += 1
        # Deliberate first-call failure for the required replanning demonstration.
        if self.primary_resource_attempts == 1:
            return ToolResult(
                False,
                "ERROR: Primary study-resource server returned HTTP 503. "
                "The requested resource is unavailable."
            )
        return ToolResult(True, f"Retrieved primary resource for '{topic}'.")

    def use_backup_resource(self, topic: str) -> ToolResult:
        return ToolResult(
            True,
            f"Retrieved backup resource for '{topic}' from the local knowledge base."
        )

    def create_schedule(self, topics: List[str]) -> ToolResult:
        if not topics:
            return ToolResult(False, "ERROR: No topics were supplied.")
        schedule = (
            "09:00-10:30  Data Structures and Algorithms\n"
            "10:45-11:30  Aptitude practice\n"
            "11:45-12:30  Core technical interview revision\n"
            "17:00-17:30  Review and error log"
        )
        return ToolResult(True, "Schedule created:\n" + schedule)

    def validate_schedule(self, schedule_text: str) -> ToolResult:
        required = ["Data Structures", "Aptitude", "technical interview"]
        missing = [x for x in required if x.lower() not in schedule_text.lower()]
        if missing:
            return ToolResult(False, "Validation failed. Missing: " + ", ".join(missing))
        return ToolResult(True, "Validation passed: all three preparation areas are included.")


SYSTEM_PROMPT = """You are SmartTask, a goal-oriented ReAct agent.
You must choose exactly ONE next action based on the goal, current plan,
completed actions, and latest tool observation.

If an action fails, explicitly notice the error and change the plan.
Do not repeat a failed action when a reasonable alternative tool exists.
Never claim a tool succeeded unless its observation confirms success.

Available actions:
- fetch_primary_resource: arguments {"topic": string}
- use_backup_resource: arguments {"topic": string}
- create_schedule: arguments {"topics": [string, ...]}
- validate_schedule: arguments {"schedule_text": string}
- finish: arguments {}

Return ONLY valid JSON:
{
  "thought": "brief decision rationale, not private chain-of-thought",
  "action": "one available action",
  "arguments": {},
  "updated_plan": ["optional revised steps"],
  "final_answer": "short answer, only for finish"
}
The thought is a concise rationale, not hidden chain-of-thought.
"""


class OllamaQwen:
    """Local LLM adapter. Sends each ReAct cycle to Ollama's /api/chat."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        if requests is None:
            raise RuntimeError("Missing dependency. Install it with: pip install requests")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def decide(self, state: AgentState, last_observation: str) -> Dict[str, Any]:
        context = {
            "goal": state.goal,
            "current_plan": state.plan,
            "completed_actions": state.completed,
            "latest_observation": last_observation,
            "replanning_already_observed": state.replanned,
            "instruction": "Choose one next action. Return JSON only."
        }
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
            ]
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=180
            )
            response.raise_for_status()
            body = response.json()
            decision = json.loads(body["message"]["content"])
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Cannot connect to Ollama at "
                f"{self.base_url}. Start the Ollama app/service, then retry."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Ollama timed out. Retry after checking the model is loaded.") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc} — {response.text}") from exc
        except (KeyError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Could not parse Qwen's JSON response: {exc}") from exc

        if not isinstance(decision, dict) or not all(
            key in decision for key in ("thought", "action", "arguments")
        ):
            raise RuntimeError(f"Qwen returned an incomplete decision: {decision}")
        return decision


class DemoLLM:
    """Scripted mock used only for a repeatable no-network demonstration."""

    def __init__(self):
        self.turn = 0

    def decide(self, state: AgentState, last_observation: str) -> Dict[str, Any]:
        self.turn += 1
        if self.turn == 1:
            return {
                "thought": "The goal requires resource gathering before schedule creation.",
                "action": "fetch_primary_resource",
                "arguments": {"topic": "placement preparation"},
                "updated_plan": [
                    "Retrieve a placement preparation resource",
                    "Create a schedule covering DSA, aptitude and interviews",
                    "Validate the schedule"
                ]
            }
        if self.turn == 2:
            return {
                "thought": "The primary source returned HTTP 503; switch to the backup source.",
                "action": "use_backup_resource",
                "arguments": {"topic": "placement preparation"},
                "updated_plan": [
                    "Use backup resource because primary source failed",
                    "Create a schedule covering DSA, aptitude and interviews",
                    "Validate the schedule"
                ]
            }
        if self.turn == 3:
            return {
                "thought": "The backup resource succeeded; create the schedule.",
                "action": "create_schedule",
                "arguments": {"topics": ["DSA", "aptitude", "technical interview"]},
                "updated_plan": ["Create the schedule", "Validate schedule coverage"]
            }
        if self.turn == 4:
            return {
                "thought": "The schedule exists; verify all required preparation areas.",
                "action": "validate_schedule",
                "arguments": {"schedule_text": last_observation},
                "updated_plan": ["Validate schedule coverage"]
            }
        return {
            "thought": "Validation passed; the requested goal is complete.",
            "action": "finish",
            "arguments": {},
            "final_answer": "Placement preparation schedule created and validated successfully."
        }


class ReActAgent:
    def __init__(self, llm, tools: Tools, max_steps: int = 10):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def dispatch(self, action: str, args: Dict[str, Any]) -> ToolResult:
        if action == "fetch_primary_resource":
            return self.tools.fetch_primary_resource(args.get("topic", ""))
        if action == "use_backup_resource":
            return self.tools.use_backup_resource(args.get("topic", ""))
        if action == "create_schedule":
            return self.tools.create_schedule(args.get("topics", []))
        if action == "validate_schedule":
            return self.tools.validate_schedule(args.get("schedule_text", ""))
        return ToolResult(False, f"ERROR: Unknown action '{action}'.")

    def execute(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)
        last_observation = "No actions have been executed yet."
        print("\nGOAL:", goal)
        print("\n--- ReAct execution trace ---")

        for step in range(1, self.max_steps + 1):
            state.steps = step
            decision = self.llm.decide(state, last_observation)
            thought = str(decision.get("thought", "No rationale supplied."))
            action = decision.get("action")
            args = decision.get("arguments") or {}
            proposed_plan = decision.get("updated_plan") or []

            print(f"\nSTEP {step}")
            print("THOUGHT:", thought)

            if proposed_plan:
                if not state.plan:
                    state.plan = proposed_plan
                    print("INITIAL PLAN:", state.plan)
                elif proposed_plan != state.plan:
                    state.plan = proposed_plan
                    state.replanned = True
                    print("REVISED PLAN:", state.plan)

            print("ACTION:", action, args)

            if action == "finish":
                last_observation = decision.get("final_answer", "Agent finished.")
                state.observations.append(last_observation)
                print("OBSERVATION:", last_observation)
                print("\nFINAL:", last_observation)
                return state

            result = self.dispatch(action, args)
            last_observation = result.output
            state.observations.append(last_observation)
            print("OBSERVATION:", last_observation)

            if result.success:
                state.completed.append(action)
            else:
                print("STATUS: FAILED — failure observation will be sent to the model.")

        raise RuntimeError(
            "Maximum step limit reached. The model did not finish; inspect the trace."
        )


def main():
    parser = argparse.ArgumentParser(description="Local Ollama + Qwen ReAct agent")
    parser.add_argument("--demo", action="store_true",
                        help="Use scripted offline decisions instead of Ollama")
    parser.add_argument("--model", default="qwen2.5:3b",
                        help="Exact model tag shown by 'ollama list'")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL")
    args = parser.parse_args()

    goal = (
        "Create a practical one-day placement preparation schedule covering DSA, "
        "aptitude and technical interview revision. Gather a study resource, create "
        "the schedule, and verify that all three areas are covered."
    )

    llm = DemoLLM() if args.demo else OllamaQwen(args.model, args.ollama_url)
    agent = ReActAgent(llm, Tools())
    state = agent.execute(goal)

    print("\n--- RUN SUMMARY ---")
    print("Steps executed:", state.steps)
    print("Successful tool actions:", state.completed)
    print("Dynamic replanning observed:", state.replanned)


if __name__ == "__main__":
    main()
