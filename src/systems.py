import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .corpus import Scene
from .embeddings import Embedder
from .llm import LLM, StructuredOutputError
from .retrieval import SCENES_HEADER, ContextBundle, HybridLongTermMemory, compose_context, render_scene
from .utils import count_tokens
from .vector_store import VectorStore

ANSWER_SYSTEM = """You answer questions about a TV series using only the context provided.
The context contains transcript scenes with ids such as S01E02_sc003 and may also contain facts, events and memories extracted from earlier episodes.
Rules:
- Use only the context. Do not use outside knowledge about the series.
- If the context does not contain the answer, set abstain to true and answer "I don't know".
- Keep the answer short: a few words or one sentence.
- In evidence, list the ids of the scenes that support the answer.
Return a JSON object."""

UNKNOWN_ANSWER = "I don't know"
ABSTENTION = re.compile(
    r"^\s*(i\s+(do\s*n[o']?t|don't|cannot|can't)\s+(know|tell|say|find)|unknown|not\s+(mentioned|stated|known|in the context)|"
    r"the context does not|there is no (information|mention)|no information|cannot be determined|n/?a\b)",
    re.IGNORECASE,
)


class AnswerOutput(BaseModel):
    answer: str
    abstain: bool = False
    evidence: list[str] = Field(default_factory=list)


def is_abstention(answer: str) -> bool:
    return not answer.strip() or bool(ABSTENTION.match(answer))


@dataclass
class SystemAnswer:
    answer: str
    abstained: bool
    cited_scene_ids: list[str]
    context_scene_ids: list[str]
    ranked_scene_ids: list[str]
    context_tokens: int
    trace: list[dict[str, Any]] = field(default_factory=list)


def generate_answer(llm: LLM, question: str, context: ContextBundle, conversation: str = "", num_ctx: int | None = None) -> AnswerOutput:
    dialogue = f"Conversation so far:\n{conversation}\n\n" if conversation else ""
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"{dialogue}Context:\n{context.text}\n\nQuestion: {question}"},
    ]
    try:
        output = llm.structured(messages, AnswerOutput, role="answer", num_ctx=num_ctx)
    except StructuredOutputError:
        return AnswerOutput(answer=UNKNOWN_ANSWER, abstain=True)
    if output.abstain or is_abstention(output.answer):
        return AnswerOutput(answer=UNKNOWN_ANSWER, abstain=True, evidence=output.evidence)
    return output


def to_system_answer(output: AnswerOutput, context: ContextBundle, trace: list[dict[str, Any]] | None = None) -> SystemAnswer:
    cited = [scene_id for scene_id in output.evidence if scene_id in context.context_scene_ids]
    return SystemAnswer(output.answer, output.abstain, cited, context.context_scene_ids, context.ranked_scene_ids, context.tokens, trace or [])


class QASystem:
    name: str

    def answer(self, question: str, cutoff: int) -> SystemAnswer:
        raise NotImplementedError

    def reset(self) -> None:
        return None


class LongContextBaseline(QASystem):
    name = "B0"

    def __init__(self, llm: LLM, scenes: list[Scene], budget_tokens: int, num_ctx: int):
        self.llm = llm
        self.scenes = sorted(scenes, key=lambda scene: scene.order_key)
        self.budget_tokens = budget_tokens
        self.num_ctx = num_ctx

    def build_context(self, cutoff: int) -> ContextBundle:
        chosen: list[str] = []
        ids: list[str] = []
        used = count_tokens(SCENES_HEADER)
        for scene in reversed([scene for scene in self.scenes if scene.episode_index <= cutoff]):
            rendered = render_scene(scene.scene_id, scene.episode_id, scene.text)
            size = count_tokens(rendered) + 1
            if used + size > self.budget_tokens:
                break
            chosen.insert(0, rendered)
            ids.insert(0, scene.scene_id)
            used += size
        text = SCENES_HEADER + "\n" + "\n\n".join(chosen)
        return ContextBundle(text, count_tokens(text), ids, ids)

    def answer(self, question: str, cutoff: int) -> SystemAnswer:
        context = self.build_context(cutoff)
        return to_system_answer(generate_answer(self.llm, question, context, num_ctx=self.num_ctx), context)


class VectorRAGBaseline(QASystem):
    name = "B1"

    def __init__(self, llm: LLM, scene_index: VectorStore, embedder: Embedder, top_k: int, context_tokens: int):
        self.llm = llm
        self.scene_index = scene_index
        self.embedder = embedder
        self.top_k = top_k
        self.context_tokens = context_tokens

    def answer(self, question: str, cutoff: int) -> SystemAnswer:
        hits = self.scene_index.search(self.embedder.embed_query(question), self.top_k, cutoff)
        context = compose_context(hits, self.context_tokens)
        return to_system_answer(generate_answer(self.llm, question, context), context)


class HybridMemorySystem(QASystem):
    name = "C1"

    def __init__(self, llm: LLM, memory: HybridLongTermMemory):
        self.llm = llm
        self.memory = memory

    def answer(self, question: str, cutoff: int) -> SystemAnswer:
        evidence = self.memory.gather(question, cutoff)
        context = self.memory.compose(evidence)
        trace = [{"node": "hybrid_retrieve", "entities": evidence.entity_ids, "sections": context.sections}]
        return to_system_answer(generate_answer(self.llm, question, context), context, trace)
