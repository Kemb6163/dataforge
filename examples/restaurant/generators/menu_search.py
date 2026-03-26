"""Menu search generator — single tool call SFT examples.

Produces ~120 examples covering:
- Keyword search
- Category filtering
- Dietary restriction filtering
- Price range queries
- Dish detail lookups
- Error handling for no-result / timeout scenarios
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import (
    user_msg, tool_call_msg, tool_result_msg, assistant_msg, reset_call_counter,
)
from dataforge.core.styles import pick_style, pick_structure, build_response, format_tool_results
from dataforge.core.errors import should_inject_error, make_error_response, make_error_handling_response
from dataforge.core.types import Example
from dataforge.generation.base import SFTGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import (
    MENU_ITEMS, ALL_DISHES, DISH_BY_ID, MENU_QUERIES,
    get_random_dish, get_random_dishes,
)


class MenuSearchGenerator(SFTGenerator):
    """Generates single-tool menu search examples."""

    @property
    def category(self) -> str:
        return "menu_search"

    @property
    def name(self) -> str:
        return "Menu Search"

    def expected_count(self) -> int:
        return 120

    def generate(self) -> Iterator[Example]:
        seed = self.config.get("seed", 42)
        error_cfg = self.config.get("error_injection", {})
        error_enabled = error_cfg.get("enabled", True)
        error_rate = error_cfg.get("base_rate", 0.10)
        custom_styles = self.config.get("styles", {})
        total = self.expected_count()

        idx = 0

        # Pattern 1: Keyword search (30 examples)
        for i in range(30):
            rng = make_rng(self.category, idx, seed)
            query = rng.choice(MENU_QUERIES)
            if "{max_price}" in query:
                query = query.replace("{max_price}", str(rng.randint(10, 30)))

            # Check error injection
            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs = [
                    user_msg(query),
                    tool_call_msg("search_menu", {"query": query}, prefix=self.category, rng=rng),
                ]
                call_id = msgs[-1]["tool_calls"][0]["id"]
                msgs.append(tool_result_msg(call_id, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            results = get_random_dishes(rng, rng.randint(1, 4))
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            if results:
                items_text = []
                for r in results:
                    dietary_str = f" ({', '.join(r['dietary'])})" if r["dietary"] else ""
                    items_text.append(f"**{r['name']}** — ${r['price']:.2f}{dietary_str}")
                body = "\n".join(f"- {t}" for t in items_text)
            else:
                body = ""

            response = build_response(body, style, structure, rng, no_result=(not results))

            msgs = [
                user_msg(query),
                tool_call_msg("search_menu", {"query": query}, prefix=self.category, rng=rng),
            ]
            call_id = msgs[-1]["tool_calls"][0]["id"]
            search_result = [{"name": r["name"], "id": r["id"], "price": r["price"], "dietary": r["dietary"]} for r in results]
            msgs.append(tool_result_msg(call_id, json.dumps(search_result)))
            msgs.append(assistant_msg(response))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Category filter (25 examples)
        categories = list(MENU_ITEMS.keys())
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            cat = rng.choice(categories)
            query_templates = [
                f"Show me your {cat}",
                f"What {cat} do you have?",
                f"I'd like to see the {cat} section",
                f"Can I browse your {cat}?",
                f"What's available in {cat}?",
            ]
            query = rng.choice(query_templates)

            results = get_random_dishes(rng, rng.randint(2, 5), cat)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            items_text = []
            for r in results:
                items_text.append(f"**{r['name']}** — ${r['price']:.2f}")
            body = f"Here are some items from our {cat} menu:\n" + "\n".join(f"- {t}" for t in items_text)
            response = build_response(body, style, structure, rng)

            msgs = [
                user_msg(query),
                tool_call_msg("search_menu", {"query": cat, "category": cat}, prefix=self.category, rng=rng),
            ]
            call_id = msgs[-1]["tool_calls"][0]["id"]
            search_result = [{"name": r["name"], "id": r["id"], "price": r["price"]} for r in results]
            msgs.append(tool_result_msg(call_id, json.dumps(search_result)))
            msgs.append(assistant_msg(response))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Dietary restriction (25 examples)
        diets = ["vegetarian", "vegan", "gluten-free", "dairy-free"]
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            diet = rng.choice(diets)
            query_templates = [
                f"What {diet} options do you have?",
                f"I need {diet} dishes please",
                f"Show me everything that's {diet}",
                f"My friend is {diet}, what can they eat?",
                f"Any {diet} recommendations?",
            ]
            query = rng.choice(query_templates)

            matching = [d for d in ALL_DISHES if diet in d.get("dietary", [])]
            results = rng.sample(matching, min(rng.randint(2, 5), len(matching))) if matching else []

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            if results:
                items_text = []
                for r in results:
                    items_text.append(f"**{r['name']}** ({r['category']}) — ${r['price']:.2f}")
                body = f"Here are our {diet} options:\n" + "\n".join(f"- {t}" for t in items_text)
            else:
                body = ""
            response = build_response(body, style, structure, rng, no_result=(not results))

            msgs = [
                user_msg(query),
                tool_call_msg("search_menu", {"query": diet, "dietary": diet}, prefix=self.category, rng=rng),
            ]
            call_id = msgs[-1]["tool_calls"][0]["id"]
            search_result = [{"name": r["name"], "id": r["id"], "price": r["price"], "category": r["category"]} for r in results]
            msgs.append(tool_result_msg(call_id, json.dumps(search_result)))
            msgs.append(assistant_msg(response))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Dish detail lookup (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            dish = get_random_dish(rng)
            query_templates = [
                f"Tell me more about the {dish['name']}",
                f"What's in the {dish['name']}?",
                f"Can I get details on the {dish['name']}?",
                f"What allergens does the {dish['name']} contain?",
                f"Is the {dish['name']} any good?",
            ]
            query = rng.choice(query_templates)

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs = [
                    user_msg(query),
                    tool_call_msg("get_dish_details", {"dish_id": dish["id"]}, prefix=self.category, rng=rng),
                ]
                call_id = msgs[-1]["tool_calls"][0]["id"]
                msgs.append(tool_result_msg(call_id, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            detail_result = {
                "name": dish["name"],
                "price": dish["price"],
                "category": dish["category"],
                "dietary": dish.get("dietary", []),
                "calories": dish.get("calories", 0),
                "description": f"Our signature {dish['name'].lower()}, prepared fresh daily.",
            }

            body_parts = [
                f"**{dish['name']}** — ${dish['price']:.2f}",
                f"Category: {dish['category'].title()}",
                f"Calories: {dish.get('calories', 'N/A')}",
            ]
            if dish.get("dietary"):
                body_parts.append(f"Dietary: {', '.join(dish['dietary'])}")
            body = "\n".join(body_parts)

            response = build_response(body, style, structure, rng)

            msgs = [
                user_msg(query),
                tool_call_msg("get_dish_details", {"dish_id": dish["id"]}, prefix=self.category, rng=rng),
            ]
            call_id = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(call_id, json.dumps(detail_result)))
            msgs.append(assistant_msg(response))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 5: Price range (15 examples)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            max_price = rng.choice([10, 15, 20, 25, 30])
            query_templates = [
                f"What can I get for under ${max_price}?",
                f"Show me dishes under ${max_price}",
                f"I have a budget of ${max_price}, what do you recommend?",
                f"Anything good under ${max_price}?",
            ]
            query = rng.choice(query_templates)

            matching = [d for d in ALL_DISHES if d["price"] <= max_price]
            results = rng.sample(matching, min(rng.randint(2, 5), len(matching))) if matching else []

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            if results:
                items_text = []
                for r in sorted(results, key=lambda x: x["price"]):
                    items_text.append(f"**{r['name']}** — ${r['price']:.2f}")
                body = f"Here are some options under ${max_price}:\n" + "\n".join(f"- {t}" for t in items_text)
            else:
                body = ""
            response = build_response(body, style, structure, rng, no_result=(not results))

            msgs = [
                user_msg(query),
                tool_call_msg("search_menu", {"query": f"under {max_price}", "max_price": max_price}, prefix=self.category, rng=rng),
            ]
            call_id = msgs[-1]["tool_calls"][0]["id"]
            search_result = [{"name": r["name"], "id": r["id"], "price": r["price"]} for r in results]
            msgs.append(tool_result_msg(call_id, json.dumps(search_result)))
            msgs.append(assistant_msg(response))

            yield Example(messages=msgs)
            idx += 1
