import json
import statistics
import sys
from pathlib import Path

from src.config import SegmentationConfig
from src.corpus import corpus_stats, ingest_directory
from src.utils import count_tokens

VARIANTS = {
    "A_size_only_40_700": SegmentationConfig(min_scene_tokens=40, max_scene_tokens=700, location_breaks=False),
    "B_location_40_700": SegmentationConfig(min_scene_tokens=40, max_scene_tokens=700, location_breaks=True),
    "C_location_80_500": SegmentationConfig(min_scene_tokens=80, max_scene_tokens=500, location_breaks=True),
    "D_location_120_900": SegmentationConfig(min_scene_tokens=120, max_scene_tokens=900, location_breaks=True),
}


def describe(name: str, config: SegmentationConfig, source: Path) -> dict:
    scenes = ingest_directory(source, "alias", config)
    sizes = sorted(count_tokens(scene.text) for scene in scenes)
    stats = corpus_stats(scenes)
    return {
        "variant": name,
        "scenes": stats["scenes"],
        "scenes_per_episode": round(stats["scenes"] / stats["episodes"], 1),
        "tokens": stats["tokens"],
        "mean_tokens": round(stats["mean_scene_tokens"]),
        "median_tokens": statistics.median(sizes),
        "p90_tokens": sizes[int(0.9 * len(sizes))],
        "with_heading": round(sum(1 for s in scenes if s.heading) / len(scenes), 3),
        "with_speakers": round(stats["scenes_with_speakers"], 3),
        "mean_speakers": round(statistics.mean(len(s.speakers) for s in scenes), 2),
    }


def main() -> None:
    source = Path(sys.argv[1])
    rows = [describe(name, config, source) for name, config in VARIANTS.items()]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
