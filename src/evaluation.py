import re
import string
import time
from collections import Counter
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .llm import LLM, StructuredOutputError, track_usage
from .questions import Question, distance_bucket, episode_distance
from .systems import QASystem
from .utils import append_jsonl, read_jsonl

JUDGE_SYSTEM = """You grade answers to questions about a TV series.
Compare the predicted answer with the reference answer. The prediction is correct if it gives the same information as the reference, even with different wording or extra harmless detail.
It is incorrect if it is wrong, misses an essential part of the reference, or contradicts it.
Return a JSON object with "correct" (true or false) and a short "reason"."""

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = str.maketrans("", "", string.punctuation)
RECALL_KS = (5, 10, 20)


def normalize_answer(text: str) -> str:
    return " ".join(ARTICLES.sub(" ", text.lower().translate(PUNCTUATION)).split())


def exact_match(prediction: str, gold: Sequence[str]) -> float:
    return float(any(normalize_answer(prediction) == normalize_answer(answer) for answer in gold))


def _f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold: Sequence[str]) -> float:
    return max(_f1(prediction, answer) for answer in gold)


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float | None:
    if not gold_ids:
        return None
    top = set(ranked_ids[:k])
    return sum(1 for scene_id in set(gold_ids) if scene_id in top) / len(set(gold_ids))


def answer_scores(prediction: str, abstained: bool, gold: Sequence[str], unanswerable: bool) -> dict[str, float]:
    if unanswerable:
        return {"em": float(abstained), "f1": float(abstained)}
    if abstained:
        return {"em": 0.0, "f1": 0.0}
    return {"em": exact_match(prediction, gold), "f1": token_f1(prediction, gold)}


def paired_bootstrap(first: Sequence[float], second: Sequence[float], resamples: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    n = len(differences)
    means = differences[np.random.default_rng(seed).integers(0, n, size=(resamples, n))].mean(axis=1)
    return float(differences.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar_exact(first: Sequence[bool], second: Sequence[bool]) -> float:
    only_first = sum(1 for a, b in zip(first, second) if a and not b)
    only_second = sum(1 for a, b in zip(first, second) if b and not a)
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(only_first, only_second) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


class JudgeVerdict(BaseModel):
    correct: bool
    reason: str = ""


def judge_answer(llm: LLM, question: Question, prediction: str, abstained: bool) -> JudgeVerdict:
    if question.type == "unanswerable":
        return JudgeVerdict(correct=abstained, reason="rule: abstention expected")
    if abstained:
        return JudgeVerdict(correct=False, reason="rule: abstained on an answerable question")
    references = "; ".join(question.gold_answers)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Question: {question.question}\nReference answer: {references}\nPredicted answer: {prediction}"},
    ]
    try:
        return llm.structured(messages, JudgeVerdict, role="judge")
    except StructuredOutputError as error:
        return JudgeVerdict(correct=False, reason=f"judge failure: {error}")


def evaluate_question(system: QASystem, question: Question, episode_index: dict[str, int]) -> tuple[dict[str, Any], list]:
    system.reset()
    with track_usage() as usage:
        started = time.perf_counter()
        error = None
        try:
            result = system.answer(question.question, episode_index[question.cutoff_episode_id])
        except Exception as exc:
            result, error = None, f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - started
    distance = episode_distance(question, episode_index)
    row: dict[str, Any] = {
        "question_id": question.id,
        "system": system.name,
        "type": question.type,
        "distance": distance,
        "distance_bucket": distance_bucket(distance),
        "error": error,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "llm_calls": len(usage.calls),
        "llm_latency_s": round(usage.latency_s, 3),
        "wall_s": round(wall, 3),
    }
    if result is not None:
        evidence = question.evidence_scene_ids
        row.update(
            {
                "answer": result.answer,
                "abstained": result.abstained,
                "cited_scene_ids": result.cited_scene_ids,
                "context_scene_ids": result.context_scene_ids,
                "ranked_scene_ids": result.ranked_scene_ids[: max(RECALL_KS)],
                "context_tokens": result.context_tokens,
                "context_recall": recall_at_k(result.context_scene_ids, evidence, len(result.context_scene_ids)),
                "trace": result.trace,
                **{f"recall@{k}": recall_at_k(result.ranked_scene_ids, evidence, k) for k in RECALL_KS},
                **answer_scores(result.answer, result.abstained, question.gold_answers, question.type == "unanswerable"),
            }
        )
    return row, usage.transcript()


def run_system(system: QASystem, questions: list[Question], episode_index: dict[str, int], run_dir: Path) -> None:
    out = run_dir / f"{system.name}.jsonl"
    done = {row["question_id"] for row in read_jsonl(out) if not row["error"]}
    pending = [q for q in questions if q.id not in done]
    print(f"[{system.name}] {len(done)} done, {len(pending)} pending")
    for position, question in enumerate(pending, start=1):
        row, calls = evaluate_question(system, question, episode_index)
        append_jsonl(out, row)
        append_jsonl(run_dir / "prompts" / f"{system.name}.jsonl", {"question_id": question.id, "calls": calls})
        status = row["error"] or f"{row['answer'][:60]!r}"
        print(f"[{system.name}] {position}/{len(pending)} {question.id} ({question.type}) {row['wall_s']:.1f}s -> {status}", flush=True)


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not row["error"] or row["question_id"] not in rows:
            rows[row["question_id"]] = row
    return rows


def judge_run(run_dir: Path, systems: list[str], questions: list[Question], llm: LLM) -> None:
    by_id = {q.id: q for q in questions}
    out = run_dir / "judgements.jsonl"
    done = {(row["system"], row["question_id"]) for row in read_jsonl(out)}
    for system in systems:
        rows = latest_rows(run_dir / f"{system}.jsonl")
        pending = [row for qid, row in rows.items() if (system, qid) not in done and not row["error"] and qid in by_id]
        print(f"[judge:{system}] {len(pending)} pending")
        for row in pending:
            verdict = judge_answer(llm, by_id[row["question_id"]], row["answer"], row["abstained"])
            append_jsonl(out, {"system": system, "question_id": row["question_id"], "correct": verdict.correct, "reason": verdict.reason})
