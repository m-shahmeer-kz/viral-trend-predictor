"""Turn a rising topic into a short-video script.

If the ``OPENAI_API_KEY`` environment variable is set (and the ``openai`` package is
installed), an LLM writes the script. Otherwise a built-in template is used, so the
project always works offline and for free.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class Script:
    hook: str
    points: list[str]
    cta: str
    source: str = "template"

    def scenes(self) -> list[str]:
        return [self.hook, *self.points, self.cta]

    def to_dict(self) -> dict:
        return asdict(self)


def template_script(topic: str, keywords: list[str], momentum: float | None = None) -> Script:
    kw = [k for k in keywords if k][:3] or [topic]
    growth = f"up {momentum:.1f}x in 2 weeks" if momentum and momentum > 1 else "trending right now"
    return Script(
        hook=f"Everyone is talking about {topic.upper()}",
        points=[
            f"It's {growth}",
            f"Top keywords: {', '.join(kw)}",
            "Here's why it's spreading",
        ],
        cta="Follow for tomorrow's trend",
        source="template",
    )


def llm_script(topic: str, keywords: list[str], momentum: float | None = None,
               model: str = "gpt-4o-mini") -> Script:
    from openai import OpenAI  # optional dependency

    client = OpenAI()
    prompt = (
        "Write a 15-second vertical short-video script as JSON with keys "
        '"hook" (max 7 words), "points" (list of 3 lines, max 7 words each), "cta" (max 6 words). '
        f"Topic: {topic}. Keywords: {', '.join(keywords)}. Momentum: {momentum}."
    )
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.8)
    data = json.loads(resp.choices[0].message.content)
    return Script(hook=data["hook"], points=list(data["points"])[:3], cta=data["cta"], source=model)


def make_script(topic: str, keywords: list[str], momentum: float | None = None) -> Script:
    if os.getenv("OPENAI_API_KEY"):
        try:
            return llm_script(topic, keywords, momentum)
        except Exception as err:  # network down, no package, bad key...
            print(f"[script] OpenAI unavailable ({err.__class__.__name__}); using template.")
    return template_script(topic, keywords, momentum)
