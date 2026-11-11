import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from pydantic import BaseModel, Field

from .corpus import Scene, episode_ids, episode_of_scene
from .embeddings import Embedder
from .llm import LLM, StructuredOutputError
from .memory import scene_header
from .utils import append_jsonl, count_tokens, read_jsonl, write_jsonl
from .vector_store import VectorStore

QUESTION_TYPES = ("fact", "cross_episode", "order", "change", "unanswerable")
DISTANCE_BUCKETS = ("same", "1-5", ">5")
UNANSWERABLE = "unanswerable"

RECAP_SYSTEM = """You write evaluation questions about a TV series from human-written episode recaps.
Each question must be specific and self-contained: a viewer who watched the series must be able to tell exactly which moment it refers to, so name the characters, places or objects involved.
Never mention recaps, episodes, seasons, actors, guest stars or anything about the production. Ask only about what happens in the story.
Answers must be short (a few words or one short sentence) and stated in the recaps. Avoid yes/no questions.
Return a JSON object."""
RECAP_FACT_TASK = "Write {n} questions about different concrete facts in this recap (who, what, where, why). Use full character names. Each answer must be stated in the recap."
RECAP_CROSS_TASK = "Write one question about {name} whose answer needs one fact from each text, for example who or what links an earlier event involving {name} to a later one. The question must name the earlier event so a viewer knows what it refers to. Also give earlier_fact and later_fact: the concrete event used from each text (what someone did, said or discovered), as one short sentence without interpretation. If that is not possible, set usable to false."
RECAP_CHANGE_TASK = "If {name}'s situation, beliefs or relationship with another character clearly changed over time between the two texts, write a question asking how it changed (name both characters and what the relationship is about) and an answer that states the situation before and after. Also give earlier_fact and later_fact: the concrete event in each text that shows the situation (what someone did, said or discovered), as one short sentence without interpretation. Otherwise set usable to false."
RECAP_ORDER_TASK = "Describe one distinctive event from each recap in a short phrase (under 12 words) that names the characters. event_a comes from the earlier recap and event_b from the later one. They must be different events. If not possible, set usable to false."
SUPPORT_SYSTEM = "You check whether a transcript scene of a TV series shows or states a given fact. Answer only from the scene. Return a JSON object."

MIN_SCENE_TOKENS = 80
MAX_ATTEMPT_FACTOR = 4
CANDIDATE_SCENES = 4
SEARCH_DEPTH = 60
MIN_RECAP_WORDS = 30
OUT_OF_STORY = re.compile(r"\b(recaps?|episodes?|seasons?|guest[- ]stars?|portray\w*|actors?|actress\w*|played by|cast|texts?|scenes?)\b", re.IGNORECASE)
GENERIC_SPEAKERS = {"Agent", "Man", "Woman", "Mr.", "Mrs.", "Dr.", "Doctor", "Guard", "Voice", "Girl", "Boy", "Officer", "Soldier", "Nurse", "All"}
REVIEW_PROMPT = "[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > "


@dataclass
class Question:
    id: str
    type: str
    question: str
    answer: str
    cutoff_episode_id: str
    evidence_scene_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    series: str = ""
    verified: bool = False
    source: str = "manual"
    notes: str = ""

    @property
    def gold_answers(self) -> list[str]:
        return [self.answer, *self.aliases]


def load_questions(path: Path, verified_only: bool = False) -> list[Question]:
    questions = [Question(**row) for row in read_jsonl(path)]
    return [q for q in questions if q.verified] if verified_only else questions


def save_questions(questions: list[Question], path: Path) -> None:
    write_jsonl(path, (asdict(q) for q in questions))


def episode_distance(question: Question, episode_index: dict[str, int]) -> int | None:
    indices = [episode_index[episode_of_scene(scene_id)] for scene_id in question.evidence_scene_ids]
    if not indices or question.type == "unanswerable":
        return None
    return episode_index[question.cutoff_episode_id] - min(indices)


def distance_bucket(distance: int | None) -> str | None:
    if distance is None:
        return None
    if distance <= 0:
        return "same"
    return "1-5" if distance <= 5 else ">5"


def stratified_sample(questions: list[Question], size: int, seed: int = 13) -> list[Question]:
    if size >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    by_type: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        by_type[question.type].append(question)
    for group in by_type.values():
        rng.shuffle(group)
    sample: list[Question] = []
    while len(sample) < size:
        for qtype in QUESTION_TYPES:
            if by_type[qtype] and len(sample) < size:
                sample.append(by_type[qtype].pop())
    chosen = {q.id for q in sample}
    return [q for q in questions if q.id in chosen]


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def answer_adds_information(question: Question) -> bool:
    if question.type in ("order", "unanswerable"):
        return True
    asked = {word[:4] for word in normalized(question.question).split()}
    answer = [word[:4] for word in normalized(question.answer).split() if len(word) > 1]
    return bool(answer) and sum(word in asked for word in answer) / len(answer) < 0.75


def in_story(*texts: str) -> bool:
    return not any(OUT_OF_STORY.search(text) for text in texts)


class EventPair(BaseModel):
    event_a: str
    event_b: str
    usable: bool = True


class RecapQA(BaseModel):
    question: str
    answer: str


class RecapQAList(BaseModel):
    items: list[RecapQA] = Field(default_factory=list)


class LinkedQA(BaseModel):
    question: str
    answer: str
    earlier_fact: str = ""
    later_fact: str = ""
    usable: bool = True


class SupportCheck(BaseModel):
    supported: bool
    reason: str = ""


def load_recaps(path: Path) -> dict[str, str]:
    return {row["episode_id"]: row["recap"] for row in read_jsonl(path) if len(row["recap"].split()) >= MIN_RECAP_WORDS}


class QuestionGenerator:
    def __init__(
        self,
        llm: LLM,
        scenes: list[Scene],
        series: str,
        recaps: dict[str, str],
        scene_index: VectorStore,
        embedder: Embedder,
        seed: int = 13,
        facts_per_recap: int = 3,
    ):
        self.llm = llm
        self.series = series
        self.recaps = recaps
        self.scene_index = scene_index
        self.embedder = embedder
        self.facts_per_recap = facts_per_recap
        self.rng = random.Random(seed)
        self.episodes = episode_ids(scenes)
        self.position = {episode: index for index, episode in enumerate(self.episodes)}
        self.by_id = {scene.scene_id: scene for scene in scenes}
        self.recap_episodes = [episode for episode in self.episodes if episode in recaps]
        self.characters = self._main_characters(scenes)
        self._fact_pool: list[tuple[str, RecapQA]] = []
        self._counter = 0

    @staticmethod
    def _main_characters(scenes: list[Scene]) -> list[str]:
        ordered = sorted(scenes, key=lambda s: s.order_key)
        counts = Counter(name.split()[0].title() for s in ordered if count_tokens(s.text) >= MIN_SCENE_TOKENS for name in s.speakers)
        return [name for name, count in counts.most_common(25) if count >= 20 and name.isalpha() and name not in GENERIC_SPEAKERS]

    def _cutoff(self, latest_evidence: int) -> str:
        last = len(self.episodes) - 1
        options = [(0, 0)]
        if latest_evidence + 1 <= last:
            options.append((1, min(5, last - latest_evidence)))
        if latest_evidence + 6 <= last:
            options.append((6, last - latest_evidence))
        low, high = self.rng.choice(options)
        return self.episodes[latest_evidence + self.rng.randint(low, high)]

    def _question(self, qtype: str, text: str, answer: str, cutoff: str, evidence: list[Scene], notes: str) -> Question:
        self._counter += 1
        return Question(
            id=f"recap-{qtype}-{self._counter:04d}",
            type=qtype,
            question=text.strip(),
            answer=answer.strip(),
            cutoff_episode_id=cutoff,
            evidence_scene_ids=[s.scene_id for s in evidence],
            series=self.series,
            source="generated:recap_future" if qtype == "unanswerable" else f"generated:recap_{qtype}",
            notes=notes,
        )

    def _ask(self, recaps: list[tuple[str, str]], task: str, output_type: type[BaseModel]):
        text = "\n\n".join(f"[{label}]\n{recap}" for label, recap in recaps)
        messages = [{"role": "system", "content": RECAP_SYSTEM}, {"role": "user", "content": f"{text}\n\nTask: {task}"}]
        try:
            return self.llm.structured(messages, output_type, role="generator")
        except StructuredOutputError:
            return None

    def _ask_pair(self, first: str, second: str, task: str, output_type: type[BaseModel]):
        return self._ask([("Earlier episode recap", self.recaps[first]), ("Later episode recap", self.recaps[second])], task, output_type)

    def _supports(self, scene: Scene, fact: str) -> bool:
        messages = [
            {"role": "system", "content": SUPPORT_SYSTEM},
            {"role": "user", "content": f"{scene_header(scene)}\n{scene.text}\n\nFact: {fact}\n\nDoes the scene show or state this fact?"},
        ]
        try:
            return self.llm.structured(messages, SupportCheck, role="verifier").supported
        except StructuredOutputError:
            return False

    def evidence_for(self, fact: str, episode_id: str) -> Scene | None:
        target = self.position[episode_id]
        hits = self.scene_index.search(self.embedder.embed_query(fact), SEARCH_DEPTH, max_episode_index=target)
        candidates = [hit.record.id for hit in hits if hit.record.episode_index == target][:CANDIDATE_SCENES]
        return next((self.by_id[scene_id] for scene_id in candidates if self._supports(self.by_id[scene_id], fact)), None)

    def _known_by(self, fact: str, cutoff: int) -> bool:
        hits = self.scene_index.search(self.embedder.embed_query(fact), CANDIDATE_SCENES, max_episode_index=cutoff)
        return any(self._supports(self.by_id[hit.record.id], fact) for hit in hits)

    def _two_recaps(self) -> tuple[str, str, str] | None:
        if not self.characters:
            return None
        name = self.rng.choice(self.characters)
        mentioning = [e for e in self.recap_episodes if name in self.recaps[e]]
        if len(mentioning) < 2:
            return None
        first = self.rng.choice(mentioning[:-1])
        later = [e for e in mentioning if self.position[e] > self.position[first]]
        return name, first, self.rng.choice(later)

    def _next_fact(self) -> tuple[str, RecapQA] | None:
        if not self._fact_pool:
            episode = self.rng.choice(self.recap_episodes)
            result = self._ask([("Recap", self.recaps[episode])], RECAP_FACT_TASK.format(n=self.facts_per_recap), RecapQAList)
            if not result:
                return None
            self._fact_pool = [(episode, item) for item in result.items if item.question.strip() and item.answer.strip() and in_story(item.question, item.answer)]
        return self._fact_pool.pop() if self._fact_pool else None

    def fact(self) -> Question | None:
        drawn = self._next_fact()
        if drawn is None:
            return None
        episode, item = drawn
        scene = self.evidence_for(f"{item.question} {item.answer}", episode)
        if scene is None:
            return None
        return self._question("fact", item.question, item.answer, self._cutoff(scene.episode_index), [scene], f"recap of {episode}")

    def _linked(self, qtype: str, task: str) -> Question | None:
        chosen = self._two_recaps()
        if chosen is None:
            return None
        name, first, second = chosen
        result = self._ask_pair(first, second, task.format(name=name), LinkedQA)
        if not result or not result.usable or not result.earlier_fact or not result.later_fact or not in_story(result.question, result.answer):
            return None
        earlier = self.evidence_for(result.earlier_fact, first)
        later = self.evidence_for(result.later_fact, second) if earlier else None
        if later is None:
            return None
        notes = f"recaps of {first} and {second}; earlier: {result.earlier_fact} | later: {result.later_fact}"
        return self._question(qtype, result.question, result.answer, self._cutoff(later.episode_index), [earlier, later], notes)

    def cross_episode(self) -> Question | None:
        return self._linked("cross_episode", RECAP_CROSS_TASK)

    def change(self) -> Question | None:
        return self._linked("change", RECAP_CHANGE_TASK)

    def order(self) -> Question | None:
        chosen = self._two_recaps()
        if chosen is None:
            return None
        _, first, second = chosen
        result = self._ask_pair(first, second, RECAP_ORDER_TASK, EventPair)
        if not result or not result.usable or not in_story(result.event_a, result.event_b):
            return None
        options = [result.event_a.strip().rstrip("."), result.event_b.strip().rstrip(".")]
        earlier = self.evidence_for(options[0], first)
        later = self.evidence_for(options[1], second) if earlier else None
        if later is None:
            return None
        shown = options[:]
        self.rng.shuffle(shown)
        text = f"Which happened first: {shown[0]}, or {shown[1]}?"
        return self._question("order", text, options[0], self._cutoff(later.episode_index), [earlier, later], f"recaps of {first} and {second}")

    def unanswerable(self) -> Question | None:
        drawn = self._next_fact()
        if drawn is None:
            return None
        episode, item = drawn
        position = self.position[episode]
        if position == 0:
            return None
        cutoff = self.episodes[self.rng.randint(max(0, position - 6), position - 1)]
        if self._known_by(f"{item.question} {item.answer}", self.position[cutoff]):
            return None
        notes = f"future probe: the recap of {episode} answers it ({item.answer}); check it is not known by {cutoff}"
        return self._question("unanswerable", item.question, UNANSWERABLE, cutoff, [], notes)

    def _collect(self, n: int, attempt: Callable[[], Question | None]) -> list[Question]:
        found: list[Question] = []
        seen: set = set()
        for _ in range(n * MAX_ATTEMPT_FACTOR):
            if len(found) >= n:
                break
            question = attempt()
            if question is None or not answer_adds_information(question):
                continue
            keys = {normalized(question.question)}
            if len(question.evidence_scene_ids) > 1:
                keys.add(tuple(sorted(question.evidence_scene_ids)))
            if keys & seen:
                continue
            seen |= keys
            found.append(question)
        return found

    def generate(self, per_type: int, types: tuple[str, ...] = QUESTION_TYPES) -> list[Question]:
        return [question for qtype in types for question in self._collect(per_type, getattr(self, qtype))]


def _edited(question: Question) -> Question:
    text = input("question (empty = keep): ").strip()
    answer = input("answer (empty = keep): ").strip()
    aliases = input("aliases separated by | (empty = keep): ").strip()
    evidence = input("extra evidence scene ids separated by | (empty = none): ").strip()
    return replace(
        question,
        question=text or question.question,
        answer=answer or question.answer,
        aliases=[a.strip() for a in aliases.split("|") if a.strip()] or question.aliases,
        evidence_scene_ids=question.evidence_scene_ids + [e.strip() for e in evidence.split("|") if e.strip()],
    )


def review_candidates(candidates_path: Path, accepted_path: Path, rejected_path: Path, scenes: list[Scene]) -> dict[str, int]:
    by_id = {scene.scene_id: scene for scene in scenes}
    done = {row["id"] for path in (accepted_path, rejected_path) for row in read_jsonl(path)}
    counts = {"accepted": 0, "rejected": 0, "skipped": 0}
    for question in load_questions(candidates_path):
        if question.id in done:
            continue
        print("=" * 80)
        print(f"{question.id} [{question.type}] cutoff={question.cutoff_episode_id}  {question.notes}")
        for scene_id in question.evidence_scene_ids:
            print(f"--- {scene_id}\n{by_id[scene_id].text}")
        print(f"Q: {question.question}\nA: {question.answer}")
        while True:
            choice = input(REVIEW_PROMPT).strip().lower()
            if choice == "q":
                return counts
            if choice == "e":
                question = _edited(question)
                print(f"Q: {question.question}\nA: {question.answer}")
            elif choice == "s":
                counts["skipped"] += 1
                break
            elif choice == "r":
                append_jsonl(rejected_path, asdict(question))
                counts["rejected"] += 1
                break
            elif choice == "a":
                append_jsonl(accepted_path, asdict(replace(question, verified=True)))
                counts["accepted"] += 1
                break
    return counts
