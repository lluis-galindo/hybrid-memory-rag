import hashlib
import json
import re

import numpy as np

from src.embeddings import Embedder
from src.llm import LLM, Completion
from src.utils import count_tokens

SPEAKER = re.compile(r"^([A-Z][\w.'\-]*(?:\s+[A-Z][\w.'\-]*){0,2}):", re.MULTILINE)
QUESTION = re.compile(r"Question: (.+)")
SCENE_ID = re.compile(r"\[(S\d{2}E\d{2}_sc\d{3}) \|")
WORD = re.compile(r"\w+", re.UNICODE)


class ScriptedLLM(LLM):
    def __init__(self, responder):
        super().__init__("scripted")
        self.responder = responder

    def _generate(self, messages, schema, options):
        text = self.responder(messages, schema)
        return Completion(text, sum(count_tokens(m["content"]) for m in messages), count_tokens(text), 0.0)


class HashingEmbedder(Embedder):
    def __init__(self, dim=512):
        super().__init__("hashing")
        self.dim = dim

    def _embed(self, texts):
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in WORD.findall(text.lower()):
                digest = hashlib.md5(word.encode("utf-8")).digest()
                matrix[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0 if digest[4] % 2 == 0 else -1.0
        return matrix


def _speakers(text):
    return list(dict.fromkeys(SPEAKER.findall(text)))


def offline_responder(messages, schema):
    title = (schema or {}).get("title", "")
    prompt = messages[-1]["content"]
    speakers = _speakers(prompt)
    if title == "SceneExtraction":
        lines = [line for line in prompt.split("\n") if SPEAKER.match(line)]
        summary = lines[0] if lines else "A scene."
        relations = [{"subject": speakers[0], "predicate": "talks_with", "object": speakers[1], "description": ""}] if len(speakers) > 1 else []
        return json.dumps(
            {
                "summary": summary,
                "entities": [{"name": s, "type": "character", "aliases": []} for s in speakers],
                "relations": relations,
                "events": [{"description": summary, "participants": speakers}],
            }
        )
    if title == "ConsolidationDecision":
        return json.dumps({"operation": "NOOP" if "talks_with" in prompt else "ADD", "target_fact_ids": re.findall(r"(F\d{6}):", prompt)[:1]})
    if title == "AnswerOutput":
        ids = SCENE_ID.findall(prompt)
        if not ids or not speakers:
            return json.dumps({"answer": "I don't know", "abstain": True, "evidence": []})
        return json.dumps({"answer": speakers[0], "abstain": False, "evidence": ids[:1]})
    if title == "RouteDecision":
        question = QUESTION.search(prompt)
        return json.dumps({"decision": "search", "search_query": question.group(1) if question else "", "reason": "offline"})
    if title == "Verification":
        return json.dumps({"verdict": "unanswerable" if "I don't know" in prompt else "supported", "search_query": "", "reason": "offline"})
    if title == "RewrittenQuestion":
        return json.dumps({"question": prompt.rsplit("Latest question:", 1)[-1].strip()})
    if title == "JudgeVerdict":
        return json.dumps({"correct": False, "reason": "offline judge"})
    if title == "RecapQAList":
        return json.dumps({"items": [{"question": "Who opens the bakery?", "answer": "Sydney"}]})
    if title == "LinkedQA":
        return json.dumps({"question": "What links both episodes?", "answer": "Sydney", "earlier_fact": "Sydney appears", "later_fact": "Sydney returns", "usable": True})
    if title == "SupportCheck":
        return json.dumps({"supported": True, "reason": "offline"})
    if title == "EventPair":
        return json.dumps({"event_a": "the first event", "event_b": "the second event", "usable": True})
    return "{}" if schema is not None else "Offline summary."
