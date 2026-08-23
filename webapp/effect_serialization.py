from __future__ import annotations

from imagemessrs.core.registry import Effect


def serialize_effects(effects: list[Effect]) -> list[dict]:
    return [
        {
            "name": e.name,
            "label": e.label,
            "category": e.category,
            "multi_image": e.multi_image,
            "accepts_mask": e.accepts_mask,
            "description": e.description,
            "about": e.about,
            "params": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "default": p.default,
                    "label": p.label,
                    "description": p.description,
                    "min": p.min,
                    "max": p.max,
                    "step": p.step,
                    "choices": p.choices,
                }
                for p in e.params
            ],
        }
        for e in effects
    ]
