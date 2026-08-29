r"""
Topic 13 — Customer Complaint Triage Graph
Multi-Node LangGraph Workflow

Graph shape:

     classify_with_consistency  (3x sampled classify calls, majority vote)
               |
      (conditional edge)
     /      |       \        \
billing technical general  needs_clarification
   |        |         |            |
  END      END       END          END

Ambiguity is detected via self-consistency: the classify prompt is run
3 times at temperature=0.7. If all 3 runs agree, that's a confident
category. If they split, that disagreement itself IS the ambiguity
signal -- no separate "are you sure?" call, which we found earlier
causes a small model to over-trigger when explicitly asked to doubt itself.

Run with: python triage_graph.py
Requires: ollama serve running in background, qwen2.5:3b pulled.
"""

from collections import Counter
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# 1. Model setup
# ---------------------------------------------------------------------------
# temperature=0 -> deterministic, used for the respond_* nodes (consistent tone)
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# temperature=0.7 -> used ONLY for the 3 sampled classification votes below.
# Sampling needs some randomness or all 3 calls would just repeat identically.
llm_sampling = ChatOllama(model="qwen2.5:3b", temperature=0.7)


# ---------------------------------------------------------------------------
# 2. Graph state
# ---------------------------------------------------------------------------
class TriageState(TypedDict):
    complaint: str      # the raw customer complaint text
    category: str       # "billing" | "technical" | "general" | "needs_clarification"
    confidence_note: str  # short note explaining the classification, referencing the complaint
    votes: str            # e.g. "billing:2, technical:1" -- the raw vote breakdown, for the trace
    response: str          # final response shown to the customer


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """You are a customer-support triage assistant.
Read the complaint below carefully, then classify it into exactly ONE category:
- billing    (payments, charges, refunds, invoices, subscription cost)
- technical  (bugs, errors, app/website not working, login issues)
- general    (feedback, praise, questions, or anything not billing/technical)

Base your decision only on the complaint text given below. Do not reuse
wording from any other example — describe what THIS specific customer said.

Respond in EXACTLY this format, nothing else:
CATEGORY: <billing|technical|general>
REASON: <one sentence that quotes or closely paraphrases the specific detail from THIS complaint that drove your decision>

Complaint: "{complaint}"
"""


def _classify_once(complaint: str, model) -> tuple[str, str]:
    """One classification call. Kept as its own function so the original
    single-shot behavior (temperature=0, one call) can be restored just by
    calling this once with `llm` instead of using classify_with_consistency."""
    prompt = CLASSIFY_PROMPT.format(complaint=complaint)
    result = model.invoke(prompt).content.strip()

    category = "general"
    reason = "Could not parse a clear reason from the model output."
    for line in result.splitlines():
        line = line.strip()
        if line.upper().startswith("CATEGORY:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in ("billing", "technical", "general"):
                category = raw
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return category, reason


def classify_with_consistency(state: TriageState) -> TriageState:
    """Runs classification 3x at temperature=0.7 and takes a majority vote.
    Agreement (3/3 or 2/1) is treated as confident. A 3-way split (rare,
    but possible with only 3 categories) is treated as ambiguous too.
    This replaces asking the model to judge its own certainty -- we
    observed that approach over-triggers on a small model."""
    votes = []
    reasons = []
    for _ in range(3):
        category, reason = _classify_once(state["complaint"], llm_sampling)
        votes.append(category)
        reasons.append(reason)

    tally = Counter(votes)
    top_category, top_count = tally.most_common(1)[0]
    vote_summary = ", ".join(f"{cat}:{count}" for cat, count in tally.most_common())

    if top_count >= 2:
        # Majority agreement (2/3 or 3/3) -> confident, use the majority category
        # and the first reason that matches it for a representative explanation.
        matching_reason = next(
            (r for c, r in zip(votes, reasons) if c == top_category), reasons[0]
        )
        return {
            **state,
            "category": top_category,
            "confidence_note": matching_reason,
            "votes": vote_summary,
        }
    else:
        # No majority (e.g. 1/1/1 split across all three categories) -> genuinely ambiguous
        return {
            **state,
            "category": "needs_clarification",
            "confidence_note": "3 independent classification attempts did not agree.",
            "votes": vote_summary,
        }


def _respond(state: TriageState, department: str, tone_hint: str) -> TriageState:
    prompt = (
        f"You are the {department} support team. Write a short (2-3 sentence) "
        f"reply to this customer complaint. {tone_hint}\n\n"
        f'Complaint: "{state["complaint"]}"'
    )
    reply = llm.invoke(prompt).content.strip()
    return {**state, "response": reply}


def respond_billing(state: TriageState) -> TriageState:
    return _respond(
        state, "Billing",
        "Acknowledge the charge/payment issue and say what you'll check next.",
    )


def respond_technical(state: TriageState) -> TriageState:
    return _respond(
        state, "Technical Support",
        "Acknowledge the bug/issue and ask for one piece of info (e.g. device, error message) to help debug.",
    )


def respond_general(state: TriageState) -> TriageState:
    return _respond(
        state, "General Support",
        "Acknowledge the message warmly and say a team member will follow up.",
    )


def needs_clarification(state: TriageState) -> TriageState:
    # Triggered when the 3 sampled classification votes didn't agree.
    prompt = (
        "This customer complaint's category was unclear even after multiple "
        "classification attempts. Write one short, friendly clarifying "
        "question (1-2 sentences) asking the customer to specify whether "
        "their issue is about billing/payment, a technical problem, or "
        "something else -- without guessing on their behalf.\n\n"
        f'Complaint: "{state["complaint"]}"'
    )
    reply = llm.invoke(prompt).content.strip()
    return {**state, "response": reply}


# ---------------------------------------------------------------------------
# 4. Conditional edge — this is the routing logic jury question #1 asks about
# ---------------------------------------------------------------------------
def route_by_category(state: TriageState) -> Literal["billing", "technical", "general", "needs_clarification"]:
    # state["category"] is guaranteed to be one of these four by classify_with_consistency()
    return state["category"]


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------
graph = StateGraph(TriageState)

graph.add_node("classify_with_consistency", classify_with_consistency)
graph.add_node("respond_billing", respond_billing)
graph.add_node("respond_technical", respond_technical)
graph.add_node("respond_general", respond_general)
graph.add_node("needs_clarification", needs_clarification)

graph.set_entry_point("classify_with_consistency")

graph.add_conditional_edges(
    "classify_with_consistency",
    route_by_category,
    {
        "billing": "respond_billing",
        "technical": "respond_technical",
        "general": "respond_general",
        "needs_clarification": "needs_clarification",
    },
)

graph.add_edge("respond_billing", END)
graph.add_edge("respond_technical", END)
graph.add_edge("respond_general", END)
graph.add_edge("needs_clarification", END)

app = graph.compile()


# ---------------------------------------------------------------------------
# 6. Test cases — 3 clean cases (one per category) + 1 ambiguous case
# ---------------------------------------------------------------------------
TEST_COMPLAINTS = [
    ("BILLING (clean)",
     "You charged me twice for my monthly subscription this cycle, please refund the duplicate charge."),
    ("TECHNICAL (clean)",
     "The app crashes every time I try to upload a profile photo on Android."),
    ("GENERAL (clean)",
     "I just wanted to say the new dashboard redesign looks great, nice work."),
    ("AMBIGUOUS (premium upgrade)",
     "I was charged for the premium upgrade but the premium features never turned on — is this a billing mistake or did it just not activate?"),
    ("AMBIGUOUS (account access)",
     "I can't get into my account and I'm not sure if it's because my payment failed or because of a login bug on your site."),
]

if __name__ == "__main__":
    for label, complaint in TEST_COMPLAINTS:
        print("=" * 70)
        print(f"CASE: {label}")
        print(f"Complaint: {complaint}")
        print("-" * 70)

        final_state = app.invoke({
            "complaint": complaint,
            "category": "",
            "confidence_note": "",
            "votes": "",
            "response": "",
        })

        print(f"Vote breakdown -> {final_state['votes']}")
        print(f"Classified as -> {final_state['category']}")
        print(f"Classifier reasoning -> {final_state['confidence_note']}")
        print(f"Routed to -> {final_state['category']}")
        print(f"Final response -> {final_state['response']}")
        print()