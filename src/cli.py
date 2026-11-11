import argparse
import json
from pathlib import Path

from .config import Config, load_config
from .corpus import corpus_stats, ingest_directory, prepare_show, save_scenes
from .evaluation import judge_run, run_system
from .memory import build_long_term_memory, build_scene_index
from .questions import QUESTION_TYPES, Question, QuestionGenerator, load_questions, load_recaps, review_candidates, save_questions, stratified_sample
from .report import build_report
from .workspace import SYSTEM_NAMES, Workspace


def _ingest(config: Config, args: argparse.Namespace) -> None:
    scenes = ingest_directory(config.paths.raw_dir / config.series, config.series, config.segmentation)
    save_scenes(scenes, config.scenes_path)
    print(f"Saved {len(scenes)} scenes to {config.scenes_path}")
    print(json.dumps(corpus_stats(scenes), indent=2))


def _prepare_summscreen(config: Config, args: argparse.Namespace) -> None:
    seasons = {int(s) for s in args.seasons.split(",")}
    manifest = prepare_show(Path(args.source), args.show, seasons, config.paths.raw_dir / config.series)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


def _index(config: Config, args: argparse.Namespace) -> None:
    workspace = Workspace(config)
    store = build_scene_index(workspace.scenes, workspace.embedder)
    store.save(config.scene_index_dir)
    print(f"Indexed {len(store)} scenes with {workspace.embedder.name} into {config.scene_index_dir}")


def _build_memory(config: Config, args: argparse.Namespace) -> None:
    workspace = Workspace(config)
    scenes = workspace.scenes
    if args.max_episodes is not None:
        scenes = [s for s in scenes if s.episode_index < args.max_episodes]
    result = build_long_term_memory(scenes, workspace.llm, workspace.embedder)
    result.save(config.graph_path, config.memory_index_dir, config.consolidation_log_path)
    summary = {**result.graph.stats(), "memories": len(result.memory_index), "operations": dict(result.consolidator.operations), "failed_scenes": len(result.failures)}
    print(json.dumps(summary, indent=2))


def _generate(config: Config, args: argparse.Namespace) -> None:
    workspace = Workspace(config)
    recaps = load_recaps(Path(args.recaps))
    generator = QuestionGenerator(workspace.llm, workspace.scenes, config.series, recaps, workspace.scene_index, workspace.embedder, seed=args.seed)
    questions = generator.generate(args.per_type, tuple(args.types.split(",")))
    save_questions(questions, Path(args.out))
    print(f"Wrote {len(questions)} candidate questions to {args.out}")


def _review(config: Config, args: argparse.Namespace) -> None:
    out = Path(args.out)
    print(review_candidates(Path(args.candidates), out, out.with_suffix(".rejected.jsonl"), Workspace(config).scenes))


def _selected_questions(args: argparse.Namespace) -> list[Question]:
    questions = load_questions(Path(args.questions), verified_only=args.verified_only)
    return stratified_sample(questions, args.pilot, seed=args.seed) if args.pilot else questions


def _run(config: Config, args: argparse.Namespace) -> None:
    workspace = Workspace(config)
    questions = _selected_questions(args)
    run_dir = config.paths.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))
    (run_dir / "question_ids.json").write_text(json.dumps([q.id for q in questions]))
    for name in args.systems.split(","):
        run_system(workspace.system(name), questions, workspace.episode_index, run_dir)


def _judge(config: Config, args: argparse.Namespace) -> None:
    judge_run(config.paths.runs_dir / args.run_id, args.systems.split(","), _selected_questions(args), Workspace(config).judge_llm)


def _report(config: Config, args: argparse.Namespace) -> None:
    run_dir = config.paths.runs_dir / args.run_id
    markdown = build_report(run_dir, _selected_questions(args))
    (run_dir / "report.md").write_text(markdown)
    print(markdown)


def _ask(config: Config, args: argparse.Namespace) -> None:
    workspace = Workspace(config)
    if args.cutoff not in workspace.episode_index:
        raise SystemExit(f"Unknown episode {args.cutoff}; available: {workspace.episodes[0]}..{workspace.episodes[-1]}")
    system = workspace.system(args.system)
    print(f"Asking {system.name} with knowledge up to {args.cutoff}. Empty line to exit.")
    while question := input("> ").strip():
        result = system.answer(question, workspace.episode_index[args.cutoff])
        print(result.answer)
        if result.cited_scene_ids:
            print(f"  evidence: {', '.join(result.cited_scene_ids)}")
        if args.trace:
            for step in result.trace:
                print(f"  {step}")


def _forget(config: Config, args: argparse.Namespace) -> None:
    workspace = Workspace(config)
    removed = workspace.long_term.forget_entity(args.name)
    workspace.graph.save(config.graph_path)
    workspace.memory_index.save(config.memory_index_dir)
    print(removed)


def _add_question_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--questions", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot", type=int, default=0, help="stratified subset size")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verified-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument("--config", default="configs/alias.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    summscreen = commands.add_parser("prepare-summscreen", help="copy one show from the untokenized SummScreen release")
    summscreen.add_argument("--source", required=True, help="directory with SummScreen_raw/fd json files")
    summscreen.add_argument("--show", required=True, help="file prefix, e.g. Alias")
    summscreen.add_argument("--seasons", required=True, help="comma separated, e.g. 1,2")
    summscreen.set_defaults(handler=_prepare_summscreen)

    commands.add_parser("ingest", help="parse episodes and segment them into scenes").set_defaults(handler=_ingest)
    commands.add_parser("index", help="embed scenes into the vector index").set_defaults(handler=_index)

    memory = commands.add_parser("build-memory", help="extract the temporal knowledge graph and episodic memories")
    memory.add_argument("--max-episodes", type=int)
    memory.set_defaults(handler=_build_memory)

    generate = commands.add_parser("generate-questions", help="question candidates grounded in episode recaps")
    generate.add_argument("--recaps", required=True, help="jsonl with human-written episode recaps")
    generate.add_argument("--out", required=True)
    generate.add_argument("--per-type", type=int, default=60)
    generate.add_argument("--seed", type=int, default=13)
    generate.add_argument("--types", default=",".join(QUESTION_TYPES), help="comma separated subset of question types")
    generate.set_defaults(handler=_generate)

    review = commands.add_parser("review", help="manually review question candidates")
    review.add_argument("--candidates", required=True)
    review.add_argument("--out", required=True)
    review.set_defaults(handler=_review)

    run = commands.add_parser("run", help="answer questions with one or more systems")
    _add_question_args(run)
    run.add_argument("--systems", default=",".join(SYSTEM_NAMES))
    run.set_defaults(handler=_run)

    judge = commands.add_parser("judge", help="grade a run with the judge model")
    _add_question_args(judge)
    judge.add_argument("--systems", default=",".join(SYSTEM_NAMES))
    judge.set_defaults(handler=_judge)

    report = commands.add_parser("report", help="metrics and hypothesis tests for a run")
    _add_question_args(report)
    report.set_defaults(handler=_report)

    ask = commands.add_parser("ask", help="interactive questions")
    ask.add_argument("--cutoff", required=True, help="last episode watched, e.g. S01E05")
    ask.add_argument("--system", default="C2", choices=SYSTEM_NAMES)
    ask.add_argument("--trace", action="store_true")
    ask.set_defaults(handler=_ask)

    forget = commands.add_parser("forget", help="delete an entity from long-term memory")
    forget.add_argument("name")
    forget.set_defaults(handler=_forget)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(load_config(Path(args.config)), args)


if __name__ == "__main__":
    main()
