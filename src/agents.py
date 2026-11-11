import operator
from typing import Annotated, Any, Literal, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .config import AgentConfig
from .llm import LLM, StructuredOutputError
from .memory import ShortTermMemory
from .retrieval import ContextBundle, Evidence, HybridLongTermMemory
from .systems import UNKNOWN_ANSWER, AnswerOutput, QASystem, SystemAnswer, generate_answer, to_system_answer
from .vector_store import Hit

REWRITE_SYSTEM = """You are the orchestrator of an assistant that answers questions about a TV series.
Rewrite the user's latest question so that it can be understood without the conversation: replace pronouns and vague references with the names of the characters, places or episodes they refer to.
If the question is already self-contained, return it unchanged. Return a JSON object."""

ROUTE_SYSTEM = """You are the critic of an assistant that answers questions about a TV series.
You see a question and the context retrieved from long-term memory. Decide whether that context is enough to answer, or whether the transcripts should be searched again.
- decision: "answer" if the context is enough, "search" if evidence is missing.
- search_query: when decision is "search", a short search query for the missing scenes (names, places, key words).
- reason: one short sentence.
Return a JSON object."""

VERIFY_SYSTEM = """You are the critic of an assistant that answers questions about a TV series.
Check a draft answer against the context.
- verdict: "supported" if the context supports the draft answer (or, for a draft of "I don't know", if the context really lacks the answer but more searching could not help);
  "unsupported" if the draft is not supported by the context but the evidence may exist in other scenes;
  "unanswerable" if the question cannot be answered from the series so far, for example because it asks about something that never happened.
- search_query: when verdict is "unsupported", a short search query for the missing evidence.
- reason: one short sentence.
Return a JSON object."""

T = TypeVar("T", bound=BaseModel)


class RewrittenQuestion(BaseModel):
    question: str


class RouteDecision(BaseModel):
    decision: Literal["answer", "search"]
    search_query: str = ""
    reason: str = ""


class Verification(BaseModel):
    verdict: Literal["supported", "unsupported", "unanswerable"]
    search_query: str = ""
    reason: str = ""


class AgentState(TypedDict, total=False):
    question: str
    cutoff: int
    query: str
    conversation: str
    evidence: Evidence
    extra_hits: list[Hit]
    context: ContextBundle
    draft: AnswerOutput
    route: str
    verdict: str
    search_query: str
    rounds: int
    trace: Annotated[list[dict[str, Any]], operator.add]


class MultiAgentSystem(QASystem):
    name = "C2"

    def __init__(self, llm: LLM, memory: HybridLongTermMemory, short_term: ShortTermMemory, config: AgentConfig):
        self.llm = llm
        self.memory = memory
        self.short_term = short_term
        self.config = config
        self.app = self._build().compile()

    def reset(self) -> None:
        self.short_term.clear()

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("orchestrator_intake", self._intake)
        graph.add_node("memory_retrieve", self._retrieve)
        graph.add_node("critic_route", self._route)
        graph.add_node("corpus_search", self._search)
        graph.add_node("orchestrator_answer", self._answer)
        graph.add_node("critic_verify", self._verify)
        graph.add_node("abstain", self._abstain)
        graph.add_node("memory_write", self._write)
        graph.add_edge(START, "orchestrator_intake")
        graph.add_edge("orchestrator_intake", "memory_retrieve")
        graph.add_edge("memory_retrieve", "critic_route")
        graph.add_conditional_edges("critic_route", lambda state: state["route"], {"search": "corpus_search", "answer": "orchestrator_answer"})
        graph.add_edge("corpus_search", "orchestrator_answer")
        graph.add_edge("orchestrator_answer", "critic_verify")
        graph.add_conditional_edges("critic_verify", self._after_verify, {"search": "corpus_search", "abstain": "abstain", "done": "memory_write"})
        graph.add_edge("abstain", "memory_write")
        graph.add_edge("memory_write", END)
        return graph

    def _ask(self, system: str, user: str, output_type: type[T], role: str, fallback: T) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.llm.structured(messages, output_type, role=role)
        except StructuredOutputError:
            return fallback

    def _intake(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        conversation = self.short_term.render()
        query = question
        if conversation:
            user = f"Conversation:\n{conversation}\n\nLatest question: {question}"
            query = self._ask(REWRITE_SYSTEM, user, RewrittenQuestion, "orchestrator", RewrittenQuestion(question=question)).question.strip() or question
        return {"query": query, "conversation": conversation, "trace": [{"node": "orchestrator_intake", "query": query}]}

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        evidence = self.memory.gather(state["query"], state["cutoff"])
        context = self.memory.compose(evidence)
        trace = {"node": "memory_retrieve", "entities": evidence.entity_ids, "graph_lines": len(evidence.graph_lines), "sections": context.sections}
        return {"evidence": evidence, "context": context, "extra_hits": [], "trace": [trace]}

    def _route(self, state: AgentState) -> dict[str, Any]:
        user = f"Context:\n{state['context'].text}\n\nQuestion: {state['query']}"
        decision = self._ask(ROUTE_SYSTEM, user, RouteDecision, "critic_route", RouteDecision(decision="answer", reason="fallback"))
        route = decision.decision if state["rounds"] < self.config.max_search_rounds else "answer"
        trace = {"node": "critic_route", "decision": route, "query": decision.search_query, "reason": decision.reason}
        return {"route": route, "search_query": decision.search_query or state["query"], "trace": [trace]}

    def _search(self, state: AgentState) -> dict[str, Any]:
        already = set(state["context"].context_scene_ids) | {h.record.id for h in state["extra_hits"]}
        new_hits = self.memory.search_scenes(state["search_query"], state["cutoff"], self.config.search_top_k, exclude_ids=already)
        extra_hits = new_hits + state["extra_hits"]
        trace = {"node": "corpus_search", "query": state["search_query"], "new_hits": [h.record.id for h in new_hits]}
        return {"extra_hits": extra_hits, "context": self.memory.compose(state["evidence"], extra_hits), "rounds": state["rounds"] + 1, "trace": [trace]}

    def _answer(self, state: AgentState) -> dict[str, Any]:
        draft = generate_answer(self.llm, state["query"], state["context"], state["conversation"])
        return {"draft": draft, "trace": [{"node": "orchestrator_answer", "answer": draft.answer, "abstain": draft.abstain}]}

    def _verify(self, state: AgentState) -> dict[str, Any]:
        user = f"Context:\n{state['context'].text}\n\nQuestion: {state['query']}\nDraft answer: {state['draft'].answer}"
        verification = self._ask(VERIFY_SYSTEM, user, Verification, "critic_verify", Verification(verdict="supported", reason="fallback"))
        update: dict[str, Any] = {
            "verdict": verification.verdict,
            "trace": [{"node": "critic_verify", "verdict": verification.verdict, "reason": verification.reason}],
        }
        if verification.search_query:
            update["search_query"] = verification.search_query
        return update

    def _after_verify(self, state: AgentState) -> str:
        verdict = state["verdict"]
        if verdict == "unanswerable":
            return "abstain"
        if verdict == "unsupported":
            return "search" if state["rounds"] < self.config.max_search_rounds else "abstain"
        return "done"

    def _abstain(self, state: AgentState) -> dict[str, Any]:
        return {"draft": AnswerOutput(answer=UNKNOWN_ANSWER, abstain=True), "trace": [{"node": "abstain", "previous": state["draft"].answer}]}

    def _write(self, state: AgentState) -> dict[str, Any]:
        self.short_term.add("user", state["question"])
        self.short_term.add("assistant", state["draft"].answer)
        return {"trace": [{"node": "memory_write", "short_term_tokens": self.short_term.tokens}]}

    def answer(self, question: str, cutoff: int) -> SystemAnswer:
        state = self.app.invoke({"question": question, "cutoff": cutoff, "rounds": 0, "trace": []}, {"recursion_limit": 40})
        return to_system_answer(state["draft"], state["context"], state["trace"])
