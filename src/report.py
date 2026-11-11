from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .evaluation import latest_rows, mcnemar_exact, paired_bootstrap
from .questions import DISTANCE_BUCKETS, QUESTION_TYPES, Question
from .utils import read_jsonl

SYSTEMS = ("B0", "B1", "C1", "C2")
H1_TYPES = ("cross_episode", "order", "change")
H2_NON_INFERIORITY_MARGIN = 0.05
H2_TOKEN_RATIO = 0.5
COMPARISON_HEADER = ["| Comparison | n | Acc first | Acc second | Diff | 95% CI (bootstrap) | McNemar p |", "|---|---|---|---|---|---|---|"]

Row = dict[str, Any]
Runs = dict[str, dict[str, Row]]


@dataclass
class Comparison:
    label: str
    n: int
    first_accuracy: float
    second_accuracy: float
    difference: float
    ci_low: float
    ci_high: float
    mcnemar_p: float


def load_run(run_dir: Path, questions: list[Question]) -> tuple[Runs, bool]:
    ids = {q.id for q in questions}
    judged = {(r["system"], r["question_id"]): bool(r["correct"]) for r in read_jsonl(run_dir / "judgements.jsonl")}
    runs: Runs = {}
    for system in SYSTEMS:
        path = run_dir / f"{system}.jsonl"
        if path.exists():
            runs[system] = {qid: row for qid, row in latest_rows(path).items() if qid in ids and not row["error"]}
            for qid, row in runs[system].items():
                row["correct"] = judged.get((system, qid), row["em"] >= 1.0)
    return runs, bool(judged)


def _mean(values: list[float | None]) -> float:
    clean = [v for v in values if v is not None]
    return mean(clean) if clean else float("nan")


def _fmt(value: float, digits: int = 3) -> str:
    return "-" if value != value else f"{value:.{digits}f}"


def compare(runs: Runs, first: str, second: str, label: str, keep: Callable[[Row], bool] = lambda r: True) -> Comparison | None:
    if first not in runs or second not in runs:
        return None
    shared = sorted(qid for qid in runs[first].keys() & runs[second].keys() if keep(runs[first][qid]))
    if not shared:
        return None
    a = [bool(runs[first][q]["correct"]) for q in shared]
    b = [bool(runs[second][q]["correct"]) for q in shared]
    difference, low, high = paired_bootstrap([float(x) for x in a], [float(x) for x in b])
    return Comparison(label, len(shared), mean(a), mean(b), difference, low, high, mcnemar_exact(a, b))


def _comparison_row(c: Comparison | None) -> str:
    if c is None:
        return "| (not available) | | | | | | |"
    return (
        f"| {c.label} | {c.n} | {_fmt(c.first_accuracy)} | {_fmt(c.second_accuracy)} | {_fmt(c.difference)} "
        f"| [{_fmt(c.ci_low)}, {_fmt(c.ci_high)}] | {_fmt(c.mcnemar_p, 4)} |"
    )


def _supported(c: Comparison | None) -> str:
    if c is None:
        return "n/a"
    return "supported" if c.ci_low > 0 else "not supported"


def _overall_row(system: str, rows: list[Row]) -> str:
    def average(key: str) -> float:
        return _mean([r.get(key) for r in rows])

    accuracy = _mean([float(r["correct"]) for r in rows])
    abstention = _mean([float(r.get("abstained", False)) for r in rows])
    cells = [
        system, len(rows), _fmt(accuracy), _fmt(average("em")), _fmt(average("f1")), _fmt(average("recall@5")), _fmt(average("recall@10")),
        _fmt(average("context_recall")), _fmt(abstention), _fmt(average("total_tokens"), 0), _fmt(average("llm_calls"), 1),
        _fmt(average("llm_latency_s"), 2), _fmt(average("wall_s"), 2),
    ]
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _breakdown(title: str, runs: Runs, key: str, values: tuple[str, ...]) -> list[str]:
    lines = ["", title, "| System | " + " | ".join(values) + " |", "|---" * (len(values) + 1) + "|"]
    for system, rows in runs.items():
        subsets = [[r for r in rows.values() if r.get(key) == value] for value in values]
        cells = [f"{_fmt(_mean([float(r['correct']) for r in subset]))} (n={len(subset)})" for subset in subsets]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return lines


def _h1(runs: Runs) -> list[str]:
    main = compare(runs, "C1", "B1", "C1 vs B1 (cross_episode, order, change)", lambda r: r["type"] in H1_TYPES)
    per_type = [compare(runs, "C1", "B1", f"C1 vs B1 ({t})", lambda r, t=t: r["type"] == t) for t in (*H1_TYPES, "fact")]
    rows = [_comparison_row(c) for c in [main, *per_type]]
    return ["", "## H1: hybrid memory beats vector RAG on cross-episode, order and change questions", *COMPARISON_HEADER, *rows, f"\nVerdict: {_supported(main)}"]


def _h2(runs: Runs) -> list[str]:
    main = compare(runs, "C1", "B0", "C1 vs B0 (all)")
    lines = ["", "## H2: hybrid memory matches long context with clearly fewer tokens", *COMPARISON_HEADER, _comparison_row(main)]
    if main:
        ratio = _mean([r["total_tokens"] for r in runs["C1"].values()]) / _mean([r["total_tokens"] for r in runs["B0"].values()])
        verdict = "supported" if main.ci_low > -H2_NON_INFERIORITY_MARGIN and ratio < H2_TOKEN_RATIO else "not supported"
        lines.append(f"\nToken ratio C1/B0: {_fmt(ratio)}. Non-inferiority margin: {H2_NON_INFERIORITY_MARGIN}. Required token ratio < {H2_TOKEN_RATIO}. Verdict: {verdict}")
    return lines


def _h3(runs: Runs) -> list[str]:
    main = compare(runs, "C2", "C1", "C2 vs C1 (all)")
    unanswerable = compare(runs, "C2", "C1", "C2 vs C1 (unanswerable)", lambda r: r["type"] == "unanswerable")
    answerable = compare(runs, "C2", "C1", "C2 vs C1 (answerable)", lambda r: r["type"] != "unanswerable")
    lines = ["", "## H3: the critic improves results, mainly through correct abstentions", *COMPARISON_HEADER]
    lines += [_comparison_row(c) for c in (main, unanswerable, answerable)]
    if main:
        extra = _mean([r["wall_s"] for r in runs["C2"].values()]) - _mean([r["wall_s"] for r in runs["C1"].values()])
        lines.append(f"\nExtra time per question (C2 - C1): {_fmt(extra, 2)} s. Verdict: {_supported(main)}")
    return lines


def _h4(runs: Runs) -> list[str]:
    buckets = [compare(runs, "C1", "B1", f"C1 vs B1 (distance {b})", lambda r, b=b: r.get("distance_bucket") == b) for b in DISTANCE_BUCKETS]
    lines = ["", "## H4: the advantage of hybrid memory grows with distance", *COMPARISON_HEADER, *(_comparison_row(c) for c in buckets)]
    if all(buckets):
        increasing = all(a.difference <= b.difference for a, b in zip(buckets, buckets[1:]))
        verdict = "supported" if increasing and buckets[-1].ci_low > 0 else "not supported"
        lines.append(f"\nDifference increases with distance: {increasing}. Verdict: {verdict}")
    else:
        lines.append("\nVerdict: insufficient data (some distance buckets have no questions)")
    return lines


def build_report(run_dir: Path, questions: list[Question]) -> str:
    runs, uses_judge = load_run(run_dir, questions)
    lines = [
        f"# Evaluation report: {run_dir.name}",
        "",
        f"Correctness source: {'LLM judge' if uses_judge else 'exact match (judge not run yet)'}",
        "",
        "## Overall",
        "| System | n | Acc | EM | F1 | R@5 | R@10 | Ctx recall | Abstain | Tokens/q | Calls/q | LLM s/q | Wall s/q |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_overall_row(system, list(rows.values())) for system, rows in runs.items()]
    lines += _breakdown("## Accuracy by question type", runs, "type", QUESTION_TYPES)
    lines += _breakdown("## Accuracy by distance (episodes between evidence and question)", runs, "distance_bucket", DISTANCE_BUCKETS)
    return "\n".join(lines + _h1(runs) + _h2(runs) + _h3(runs) + _h4(runs)) + "\n"
