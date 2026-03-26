"""Reviews generator — no-tool restraint + review submission SFT examples.

Produces ~100 examples covering:
- Review submission with rating
- Out-of-scope request handling (no-tool restraint)
- Polite refusals for off-topic requests
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import user_msg, tool_call_msg, tool_result_msg, assistant_msg
from dataforge.core.styles import pick_style, pick_structure, build_response
from dataforge.core.errors import should_inject_error, make_error_response, make_error_handling_response
from dataforge.core.types import Example
from dataforge.generation.base import SFTGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import (
    get_order_id, get_random_dish, OUT_OF_SCOPE_REQUESTS, REFUSAL_TEMPLATES,
    REVIEW_ADJECTIVES, EXPERIENCE_ADJECTIVES, SERVICE_QUALITIES,
)


_REFUSAL_TOPICS = {
    "taxi": "a ride-hailing app",
    "weather": "a weather service",
    "movie": "a movie review site",
    "mall": "the mall's website",
    "homework": "an educational platform",
    "game": "a sports news site",
    "translate": "a translation service",
    "stock": "a financial platform",
    "reminder": "your phone's reminder app",
    "hotel": "a travel booking site",
    "music": "a music streaming app",
    "airport": "a maps application",
    "flight": "an airline booking site",
    "population": "a search engine",
    "poem": "a creative writing tool",
    "news": "a news aggregator",
    "doctor": "a healthcare directory",
    "computer": "a tech support service",
    "Uber": "the Uber app",
    "exchange": "a currency converter",
}


class ReviewGenerator(SFTGenerator):
    """Generates review submission and no-tool restraint examples."""

    @property
    def category(self) -> str:
        return "reviews"

    @property
    def name(self) -> str:
        return "Reviews & Restraint"

    def expected_count(self) -> int:
        return 100

    def generate(self) -> Iterator[Example]:
        seed = self.config.get("seed", 42)
        error_cfg = self.config.get("error_injection", {})
        error_enabled = error_cfg.get("enabled", True)
        error_rate = error_cfg.get("base_rate", 0.10)
        custom_styles = self.config.get("styles", {})
        total = self.expected_count()
        idx = 0

        # Pattern 1: Review submission (40 examples)
        for i in range(40):
            rng = make_rng(self.category, idx, seed)
            rating = rng.randint(1, 5)
            dish = get_random_dish(rng)
            adj = rng.choice(REVIEW_ADJECTIVES)
            exp_adj = rng.choice(EXPERIENCE_ADJECTIVES)
            service = rng.choice(SERVICE_QUALITIES)

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            # Generate review text
            review_templates = [
                f"The {dish['name']} was {adj}. Great atmosphere and {service} service.",
                f"Had a {exp_adj} dinner. The {dish['name']} stood out. Service was {service}.",
                f"I'd rate my experience {rating}/5. The {dish['name']} was {adj} and the staff was {service}.",
                f"Overall {exp_adj} experience. Highlight was the {dish['name']}. Would recommend!",
                f"The food was {adj}, especially the {dish['name']}. The service could be described as {service}.",
            ]
            review_text = rng.choice(review_templates)

            queries = [
                f"I'd like to leave a review. {rating} stars. {review_text}",
                f"Can I submit a review? Rating: {rating}. {review_text}",
                f"Here's my review: {review_text} I give it {rating} stars.",
                f"I want to rate my experience: {rating}/5. {review_text}",
            ]

            order_id = get_order_id(rng) if rng.random() > 0.5 else None

            msgs = [user_msg(rng.choice(queries))]

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                args = {"rating": rating, "review_text": review_text}
                if order_id:
                    args["order_id"] = order_id
                msgs.append(tool_call_msg("submit_review", args, prefix=self.category, rng=rng))
                call_id = msgs[-1]["tool_calls"][0]["id"]
                msgs.append(tool_result_msg(call_id, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            args = {"rating": rating, "review_text": review_text}
            if order_id:
                args["order_id"] = order_id

            msgs.append(tool_call_msg("submit_review", args, prefix=self.category, rng=rng))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            result = {
                "review_id": f"REV-{rng.randint(10000, 99999)}",
                "rating": rating,
                "status": "published",
            }
            msgs.append(tool_result_msg(call_id, json.dumps(result)))

            body = (
                f"Thank you for your review! Your {rating}-star rating has been submitted "
                f"(Review ID: {result['review_id']}). We appreciate your feedback!"
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Out-of-scope requests — no tool calls (60 examples)
        for i in range(60):
            rng = make_rng(self.category, idx, seed)
            request = OUT_OF_SCOPE_REQUESTS[i % len(OUT_OF_SCOPE_REQUESTS)]

            # Find matching topic for suggestion
            suggestion = "an appropriate service"
            topic = "that"
            for keyword, sug in _REFUSAL_TOPICS.items():
                if keyword.lower() in request.lower():
                    suggestion = sug
                    topic = keyword.lower()
                    break

            template = rng.choice(REFUSAL_TEMPLATES)
            response = template.format(topic=topic, suggestion=suggestion)

            msgs = [user_msg(request)]
            msgs.append(assistant_msg(response))

            yield Example(messages=msgs)
            idx += 1
