import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from src.corpus import load_scenes
from src.llm import LLM, StructuredOutputError, track_usage
from src.memory import SceneExtraction, extract_scene
from src.utils import count_tokens

MODELS = sys.argv[2].split(",")
SAMPLE_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def loaded_size_gb() -> str:
    output = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.strip().splitlines()
    return " | ".join(line.split()[0] + " " + " ".join(line.split()[2:4]) for line in output[1:])


def benchmark(model: str, scenes) -> dict:
    llm = LLM(model, options={"temperature": 0.0, "num_ctx": 8192, "seed": 7})
    extract_scene(llm, scenes[0])
    rows = []
    with track_usage() as usage:
        for scene in scenes:
            started = time.perf_counter()
            try:
                result: SceneExtraction | None = extract_scene(llm, scene)
            except StructuredOutputError:
                result = None
            rows.append((time.perf_counter() - started, result))
            print(f"  {model} {scene.scene_id} {rows[-1][0]:.1f}s valid={result is not None}", flush=True)
    return {
        "model": model,
        "valid": sum(1 for _, r in rows if r is not None),
        "scenes": len(rows),
        "mean_latency_s": round(statistics.mean(t for t, _ in rows), 1),
        "llm_calls": len(usage.calls),
        "output_tokens_per_s": round(usage.completion_tokens / usage.latency_s, 1),
        "mean_entities": round(statistics.mean(len(r.entities) for _, r in rows if r), 1),
        "mean_relations": round(statistics.mean(len(r.relations) for _, r in rows if r), 1),
        "mean_events": round(statistics.mean(len(r.events) for _, r in rows if r), 1),
        "loaded": loaded_size_gb(),
        "examples": [r.model_dump() for _, r in rows[:2] if r],
    }


def main() -> None:
    scenes = load_scenes(Path(sys.argv[1]))
    candidates = [s for s in scenes if 250 <= count_tokens(s.text) <= 600]
    sample = random.Random(11).sample(candidates, SAMPLE_SIZE)
    results = []
    for model in MODELS:
        subprocess.run(["ollama", "stop", model], capture_output=True)
        result = benchmark(model, sample)
        subprocess.run(["ollama", "stop", model], capture_output=True)
        print(json.dumps({k: v for k, v in result.items() if k != "examples"}), flush=True)
        results.append(result)
    Path(sys.argv[4] if len(sys.argv) > 4 else "model_benchmark.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
