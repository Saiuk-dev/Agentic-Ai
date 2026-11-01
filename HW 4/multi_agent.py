"""
PlacementPrep: A role-based multi-agent collaboration demo.
Framework: Lightweight custom supervisor with local Ollama (Qwen2.5:3B).
No paid API key required.

Run:
  pip install -r requirements.txt
  ollama pull qwen2.5:3b
  python multi_agent.py              # live local Ollama collaboration
  python multi_agent.py --demo       # deterministic demonstration, no Ollama required
"""
import argparse
import json
import re
import requests
from typing import Dict, Any

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_URL = "http://localhost:11434/api/chat"

TASK = (
    "Create a realistic one-day placement preparation plan for a student. "
    "The plan must cover DSA, aptitude, and core technical interview revision, "
    "include breaks, and finish with a short review."
)

def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output, including fenced responses."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"Model did not return JSON: {text[:500]}")
        return json.loads(match.group(0))

class OllamaAgent:
    def __init__(self, name: str, role: str, guardrail: str, model: str, url: str):
        self.name, self.role, self.guardrail = name, role, guardrail
        self.model, self.url = model, url

    def think(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            f"You are {self.name}, a role-based agent. Role: {self.role}. "
            f"Strict boundary: {self.guardrail} "
            "Collaborate with other agents; do not invent facts. "
            "Return ONLY a valid JSON object, no markdown."
        )
        user = (
            f"Shared task: {task}\nContext from other agents:\n"
            f"{json.dumps(context, ensure_ascii=False)}\n\n"
            "Return JSON with keys: decision (string), message (string), "
            "schedule (array of strings, if relevant), issues (array of strings), "
            "approved (boolean, only for evaluator)."
        )
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return extract_json(content)

class DemoAgent:
    """Deterministic agent responses to make the failure/recovery trace reproducible."""
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def think(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        if self.name == "Planner":
            if self.calls == 1:
                return {
                    "decision": "Draft an initial schedule",
                    "message": "Initial draft prioritises DSA and technical revision but accidentally omits aptitude.",
                    "schedule": [
                        "09:00-11:00 DSA problem solving",
                        "11:15-12:45 Core technical revision",
                        "14:00-16:00 Advanced DSA practice",
                        "16:15-16:45 Review"
                    ],
                    "issues": ["Aptitude practice is missing", "DSA time is disproportionately high"]
                }
            return {
                "decision": "Revise schedule using peer feedback",
                "message": "Added aptitude and balanced the time blocks with breaks.",
                "schedule": [
                    "09:00-10:15 DSA concepts and problems",
                    "10:15-10:30 Break",
                    "10:30-11:15 Aptitude practice",
                    "11:15-11:30 Break",
                    "11:30-12:30 Core technical interview revision",
                    "17:00-17:20 Review and error log"
                ],
                "issues": []
            }
        if self.name == "Resource":
            schedule = context.get("schedule", [])
            return {
                "decision": "Provide resources and audit topic coverage",
                "message": "Suggested Striver A2Z for DSA, IndiaBix for aptitude practice, and a DBMS/OOP/OS/CN revision checklist. The first draft omitted aptitude; the revised draft covers all three domains.",
                "resources": [
                    "DSA: Striver A2Z DSA Sheet",
                    "Aptitude: IndiaBix quantitative aptitude practice",
                    "Technical: DBMS, OOP, Operating Systems and Computer Networks notes"
                ],
                "issues": ["Aptitude is missing from the first draft"] if not any("Aptitude" in s for s in schedule) else [],
                "coverage_ok": any("Aptitude" in s for s in schedule)
                    and any("DSA" in s for s in schedule)
                    and any("technical" in s.lower() for s in schedule)
            }
        if self.name == "Evaluator":
            schedule = context.get("schedule", [])
            has_dsa = any("DSA" in s for s in schedule)
            has_apt = any("aptitude" in s.lower() for s in schedule)
            has_tech = any("technical" in s.lower() for s in schedule)
            has_break = any("break" in s.lower() for s in schedule)
            approved = has_dsa and has_apt and has_tech and has_break
            return {
                "decision": "Approve" if approved else "Reject and request revision",
                "message": (
                    "The revised schedule covers all three required areas and includes breaks."
                    if approved else
                    "Rejecting the draft: it does not cover all three required areas and/or lacks breaks. Planner must revise it."
                ),
                "approved": approved,
                "issues": [] if approved else [
                    name for name, ok in [
                        ("DSA", has_dsa), ("Aptitude", has_apt),
                        ("Technical interview revision", has_tech), ("Breaks", has_break)
                    ] if not ok
                ]
            }
        raise ValueError("Unknown demo agent")

class Supervisor:
    def __init__(self, agents: Dict[str, Any], demo: bool = False):
        self.agents, self.demo = agents, demo

    def run(self):
        print("=" * 72)
        print("PLACEMENTPREP | MULTI-AGENT COLLABORATION")
        print("SHARED TASK:", TASK)
        print("=" * 72)

        print("\n[SUPERVISOR] Step 1: Planner creates the first draft.")
        plan1 = self.agents["planner"].think(TASK, {})
        schedule = plan1.get("schedule", [])
        print("[PLANNER] THOUGHT/DECISION:", plan1.get("decision"))
        print("[PLANNER] PROPOSAL:")
        for item in schedule:
            print("  -", item)
        print("[PLANNER] SELF-REPORTED ISSUES:", plan1.get("issues", []))

        print("\n[SUPERVISOR] Step 2: Resource agent reviews coverage and recommends resources.")
        resource = self.agents["resource"].think(TASK, {"schedule": schedule})
        print("[RESOURCE] DECISION:", resource.get("decision"))
        print("[RESOURCE] MESSAGE:", resource.get("message"))
        for item in resource.get("resources", []):
            print("  RESOURCE:", item)
        print("[RESOURCE] ISSUES:", resource.get("issues", []))

        print("\n[SUPERVISOR] Step 3: Evaluator independently checks the proposal.")
        evaluation = self.agents["evaluator"].think(
            TASK, {"schedule": schedule, "resource_review": resource}
        )
        print("[EVALUATOR] DECISION:", evaluation.get("decision"))
        print("[EVALUATOR] MESSAGE:", evaluation.get("message"))
        print("[EVALUATOR] ISSUES:", evaluation.get("issues", []))

        if not evaluation.get("approved", False):
            print("\n[SUPERVISOR] NEGOTIATION: Evaluator rejected the draft.")
            print("[SUPERVISOR] Sending evaluator feedback to Planner for revision.")
            revised = self.agents["planner"].think(
                TASK,
                {
                    "previous_schedule": schedule,
                    "resource_feedback": resource,
                    "evaluator_feedback": evaluation,
                    "instruction": "Revise the schedule. Cover DSA, aptitude, technical revision, and include breaks."
                },
            )
            schedule = revised.get("schedule", [])
            print("[PLANNER] REVISED DECISION:", revised.get("decision"))
            print("[PLANNER] REVISED SCHEDULE:")
            for item in schedule:
                print("  -", item)

            print("\n[SUPERVISOR] Step 4: Resource agent re-checks revised plan.")
            resource2 = self.agents["resource"].think(TASK, {"schedule": schedule})
            print("[RESOURCE] RE-CHECK:", resource2.get("message"))
            print("[RESOURCE] COVERAGE OK:", resource2.get("coverage_ok"))

            print("\n[SUPERVISOR] Step 5: Evaluator performs final approval.")
            evaluation2 = self.agents["evaluator"].think(
                TASK, {"schedule": schedule, "resource_review": resource2}
            )
            print("[EVALUATOR] FINAL DECISION:", evaluation2.get("decision"))
            print("[EVALUATOR] FINAL MESSAGE:", evaluation2.get("message"))
            if not evaluation2.get("approved", False):
                raise RuntimeError("The agents could not reach an approved plan. Inspect model output.")
        else:
            resource2 = resource
            evaluation2 = evaluation

        print("\n" + "=" * 72)
        print("FINAL APPROVED SCHEDULE")
        for item in schedule:
            print("  ", item)
        print("\nRESOURCES")
        for item in resource2.get("resources", []):
            print("  ", item)
        print("\nFINAL STATUS: APPROVED")
        print("=" * 72)

def build_agents(demo: bool, model: str, url: str):
    if demo:
        return {
            "planner": DemoAgent("Planner"),
            "resource": DemoAgent("Resource"),
            "evaluator": DemoAgent("Evaluator"),
        }
    return {
        "planner": OllamaAgent(
            "Planner Agent", "Break the shared goal into a balanced, time-blocked schedule.",
            "You may propose schedules, but you must not approve your own work or claim external verification.",
            model, url),
        "resource": OllamaAgent(
            "Resource Agent", "Recommend learning resources and audit whether each required topic appears.",
            "You must not change the schedule or approve the final result; report missing coverage honestly.",
            model, url),
        "evaluator": OllamaAgent(
            "Evaluator Agent", "Independently verify coverage, breaks and realism; accept or reject with reasons.",
            "You must not invent resources or silently approve a schedule with missing required areas.",
            model, url),
    }

def main():
    parser = argparse.ArgumentParser(description="PlacementPrep multi-agent system")
    parser.add_argument("--demo", action="store_true", help="Run deterministic demo without Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_URL)
    args = parser.parse_args()
    agents = build_agents(args.demo, args.model, args.ollama_url)
    try:
        Supervisor(agents, demo=args.demo).run()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to Ollama. Start Ollama and check: ollama list")
        raise SystemExit(1)
    except requests.exceptions.HTTPError as exc:
        print("\nERROR: Ollama returned an HTTP error:", exc)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
