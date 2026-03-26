"""Complex scenario generator — multi-tool chain SFT examples.

Produces ~130 examples covering:
- Search → Detail → Reserve flow
- Search → Compare → Order
- Multi-step discovery (dietary + category + price)
- Full experience: search, reserve, order, review
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import (
    user_msg, tool_call_msg, tool_result_msg, assistant_msg, multi_tool_call_msg,
)
from dataforge.core.styles import pick_style, pick_structure, build_response
from dataforge.core.errors import should_inject_error, make_error_response, make_error_handling_response
from dataforge.core.types import Example
from dataforge.generation.base import SFTGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import (
    MENU_ITEMS, ALL_DISHES, get_random_dish, get_random_dishes,
    get_customer_name, get_reservation_date, get_reservation_time,
    get_party_size, get_order_id,
)


class ComplexScenarioGenerator(SFTGenerator):
    """Generates multi-tool chain examples."""

    @property
    def category(self) -> str:
        return "complex_scenarios"

    @property
    def name(self) -> str:
        return "Complex Scenarios"

    def expected_count(self) -> int:
        return 130

    def generate(self) -> Iterator[Example]:
        seed = self.config.get("seed", 42)
        error_cfg = self.config.get("error_injection", {})
        error_enabled = error_cfg.get("enabled", True)
        error_rate = error_cfg.get("base_rate", 0.10)
        custom_styles = self.config.get("styles", {})
        total = self.expected_count()
        idx = 0

        # Pattern 1: Search → Detail → Book (35 examples)
        for i in range(35):
            rng = make_rng(self.category, idx, seed)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            cat = rng.choice(list(MENU_ITEMS.keys()))
            cust = get_customer_name(rng)
            date = get_reservation_date(rng)
            time = get_reservation_time(rng)
            party = get_party_size(rng)

            # Turn 1: Search
            msgs = [user_msg(f"I'm looking for a nice {cat} dish. We're coming for dinner on {date}.")]

            results = get_random_dishes(rng, 3, cat)
            msgs.append(tool_call_msg("search_menu", {"query": cat, "category": cat}, prefix=self.category, rng=rng))
            cid1 = msgs[-1]["tool_calls"][0]["id"]
            search_res = [{"name": r["name"], "id": r["id"], "price": r["price"]} for r in results]
            msgs.append(tool_result_msg(cid1, json.dumps(search_res)))

            items_text = "\n".join(f"- **{r['name']}** — ${r['price']:.2f}" for r in results)
            body1 = f"Here are some {cat} options:\n{items_text}\nWould you like more details on any of these?"
            msgs.append(assistant_msg(build_response(body1, style, structure, rng)))

            # Turn 2: Get details
            chosen = rng.choice(results)
            msgs.append(user_msg(f"Tell me more about the {chosen['name']}."))

            msgs.append(tool_call_msg("get_dish_details", {"dish_id": chosen["id"]}, prefix=self.category, rng=rng))
            cid2 = msgs[-1]["tool_calls"][0]["id"]
            detail = {
                "name": chosen["name"], "price": chosen["price"],
                "calories": rng.randint(300, 800),
                "dietary": chosen.get("dietary", []),
                "description": f"Our signature {chosen['name'].lower()}, made with fresh seasonal ingredients.",
            }
            msgs.append(tool_result_msg(cid2, json.dumps(detail)))

            body2 = (
                f"**{chosen['name']}** — ${chosen['price']:.2f}\n"
                f"{detail['description']}\n"
                f"Calories: {detail['calories']}"
            )
            if detail["dietary"]:
                body2 += f"\nDietary: {', '.join(detail['dietary'])}"
            body2 += "\n\nWould you like to make a reservation?"
            msgs.append(assistant_msg(build_response(body2, style, structure, rng)))

            # Turn 3: Book
            msgs.append(user_msg(f"Yes, book for {party} at {time}. Name: {cust}."))

            msgs.append(tool_call_msg(
                "make_reservation",
                {"date": date, "time": time, "party_size": party, "name": cust},
                prefix=self.category, rng=rng,
            ))
            cid3 = msgs[-1]["tool_calls"][0]["id"]
            booking = {"confirmation_id": f"RES-{rng.randint(10000, 99999)}", "date": date, "time": time}
            msgs.append(tool_result_msg(cid3, json.dumps(booking)))

            body3 = (
                f"You're all set! Reservation confirmed:\n"
                f"- Confirmation: {booking['confirmation_id']}\n"
                f"- {date} at {time}, party of {party}\n"
                f"- Name: {cust}\n"
                f"Enjoy the {chosen['name']}!"
            )
            msgs.append(assistant_msg(build_response(body3, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Compare two dishes (parallel detail lookups) (30 examples)
        for i in range(30):
            rng = make_rng(self.category, idx, seed)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            dishes = get_random_dishes(rng, 2)

            msgs = [user_msg(f"Can you compare the {dishes[0]['name']} and the {dishes[1]['name']}?")]

            # Parallel tool calls
            calls = [("get_dish_details", {"dish_id": d["id"]}) for d in dishes]
            tc_msg = multi_tool_call_msg(calls, prefix=self.category, rng=rng)
            msgs.append(tc_msg)

            details = []
            for j, d in enumerate(dishes):
                cid = tc_msg["tool_calls"][j]["id"]
                detail = {
                    "name": d["name"], "price": d["price"],
                    "calories": d.get("calories", rng.randint(300, 800)),
                    "category": d.get("category", "mains"),
                    "dietary": d.get("dietary", []),
                }
                msgs.append(tool_result_msg(cid, json.dumps(detail)))
                details.append(detail)

            d1, d2 = details
            body = (
                f"Here's a comparison:\n\n"
                f"**{d1['name']}** — ${d1['price']:.2f}\n"
                f"- Category: {d1['category'].title()}\n"
                f"- Calories: {d1['calories']}\n"
                f"- Dietary: {', '.join(d1['dietary']) or 'None'}\n\n"
                f"**{d2['name']}** — ${d2['price']:.2f}\n"
                f"- Category: {d2['category'].title()}\n"
                f"- Calories: {d2['calories']}\n"
                f"- Dietary: {', '.join(d2['dietary']) or 'None'}"
            )

            price_diff = abs(d1["price"] - d2["price"])
            if price_diff > 5:
                cheaper = d1["name"] if d1["price"] < d2["price"] else d2["name"]
                body += f"\n\nThe {cheaper} is the more budget-friendly option (${price_diff:.2f} less)."

            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Multi-step dietary discovery (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            diet = rng.choice(["vegetarian", "vegan", "gluten-free"])
            max_price = rng.choice([15, 20, 25])

            msgs = [user_msg(f"I need a {diet} option under ${max_price}. What do you have?")]

            # Search with filters
            msgs.append(tool_call_msg(
                "search_menu",
                {"query": diet, "dietary": diet, "max_price": max_price},
                prefix=self.category, rng=rng,
            ))
            cid1 = msgs[-1]["tool_calls"][0]["id"]

            matching = [d for d in ALL_DISHES if diet in d.get("dietary", []) and d["price"] <= max_price]
            results = rng.sample(matching, min(3, len(matching))) if matching else []

            search_res = [{"name": r["name"], "id": r["id"], "price": r["price"]} for r in results]
            msgs.append(tool_result_msg(cid1, json.dumps(search_res)))

            if not results:
                body1 = f"I couldn't find any {diet} options under ${max_price}. Would you like me to expand the search?"
                msgs.append(assistant_msg(build_response(body1, style, structure, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            items_text = "\n".join(f"- **{r['name']}** — ${r['price']:.2f}" for r in results)
            body1 = f"Found {len(results)} {diet} options under ${max_price}:\n{items_text}\nWant details on any?"
            msgs.append(assistant_msg(build_response(body1, style, structure, rng)))

            # Follow-up: get details on one
            pick = rng.choice(results)
            msgs.append(user_msg(f"What about the {pick['name']}?"))

            msgs.append(tool_call_msg("get_dish_details", {"dish_id": pick["id"]}, prefix=self.category, rng=rng))
            cid2 = msgs[-1]["tool_calls"][0]["id"]
            detail = {
                "name": pick["name"], "price": pick["price"],
                "calories": pick.get("calories", rng.randint(200, 600)),
                "dietary": pick.get("dietary", []),
                "description": f"A delightful {pick['name'].lower()} prepared with care.",
            }
            msgs.append(tool_result_msg(cid2, json.dumps(detail)))

            body2 = (
                f"**{pick['name']}** — ${pick['price']:.2f}\n"
                f"{detail['description']}\n"
                f"Calories: {detail['calories']} | Dietary: {', '.join(detail['dietary'])}"
            )
            msgs.append(assistant_msg(build_response(body2, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Check availability + Search menu simultaneously (20 examples)
        for i in range(20):
            rng = make_rng(self.category, idx, seed)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            date = get_reservation_date(rng)
            time = get_reservation_time(rng)
            party = get_party_size(rng)
            cat = rng.choice(["appetizers", "mains", "desserts"])

            msgs = [user_msg(
                f"We're {party} people thinking of coming {date} at {time}. "
                f"Is that available? Also, what {cat} do you have?"
            )]

            # Parallel: check_availability + search_menu
            calls = [
                ("check_availability", {"date": date, "time": time, "party_size": party}),
                ("search_menu", {"query": cat, "category": cat}),
            ]
            msgs.append(multi_tool_call_msg(calls, prefix=self.category, rng=rng))

            avail_cid = msgs[-1]["tool_calls"][0]["id"]
            search_cid = msgs[-1]["tool_calls"][1]["id"]

            available = rng.random() > 0.25
            msgs.append(tool_result_msg(avail_cid, json.dumps({
                "available": available, "date": date, "time": time, "party_size": party,
            })))

            menu_results = get_random_dishes(rng, 3, cat)
            search_res = [{"name": r["name"], "id": r["id"], "price": r["price"]} for r in menu_results]
            msgs.append(tool_result_msg(search_cid, json.dumps(search_res)))

            avail_text = (
                f"a table for {party} is available on {date} at {time}"
                if available else
                f"we're unfortunately fully booked for {party} at {time} on {date}"
            )
            items_text = "\n".join(f"- **{r['name']}** — ${r['price']:.2f}" for r in menu_results)
            body = (
                f"I checked both things for you!\n\n"
                f"**Availability:** {avail_text.capitalize()}.\n\n"
                f"**{cat.title()} menu:**\n{items_text}"
            )
            if available:
                body += "\n\nWould you like me to book the table?"
            else:
                body += "\n\nWould you like me to check a different time?"

            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 5: Full experience chain (20 examples)
        for i in range(20):
            rng = make_rng(self.category, idx, seed)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            cust = get_customer_name(rng)
            date = get_reservation_date(rng)
            time = get_reservation_time(rng)
            party = get_party_size(rng)

            # Turn 1: Search
            msgs = [user_msg("What are your best main courses?")]
            results = get_random_dishes(rng, 3, "mains")
            msgs.append(tool_call_msg("search_menu", {"query": "best mains", "category": "mains"}, prefix=self.category, rng=rng))
            cid1 = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid1, json.dumps([{"name": r["name"], "id": r["id"], "price": r["price"]} for r in results])))

            body1 = "Our top main courses:\n" + "\n".join(f"- **{r['name']}** — ${r['price']:.2f}" for r in results)
            msgs.append(assistant_msg(build_response(body1, style, structure, rng)))

            # Turn 2: Book
            msgs.append(user_msg(f"Looks great! Book a table for {party} on {date} at {time}, name {cust}."))
            msgs.append(tool_call_msg(
                "make_reservation",
                {"date": date, "time": time, "party_size": party, "name": cust},
                prefix=self.category, rng=rng,
            ))
            cid2 = msgs[-1]["tool_calls"][0]["id"]
            booking = {"confirmation_id": f"RES-{rng.randint(10000, 99999)}", "date": date, "time": time}
            msgs.append(tool_result_msg(cid2, json.dumps(booking)))

            body2 = f"Booked! Confirmation: {booking['confirmation_id']}. {date} at {time}, party of {party}."
            msgs.append(assistant_msg(build_response(body2, style, structure, rng)))

            # Turn 3: Check order (simulating after dinner)
            order_id = get_order_id(rng)
            msgs.append(user_msg(f"We're here now, order is {order_id}. How long?"))
            msgs.append(tool_call_msg("get_order_status", {"order_id": order_id}, prefix=self.category, rng=rng))
            cid3 = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid3, json.dumps({
                "order_id": order_id, "status": "cooking", "eta_minutes": 12,
            })))
            msgs.append(assistant_msg("Your order is being cooked now. About 12 minutes until it's ready!"))

            # Turn 4: Submit review
            rating = rng.randint(3, 5)
            msgs.append(user_msg(f"Great meal! {rating} stars. Everything was delicious."))
            msgs.append(tool_call_msg(
                "submit_review",
                {"rating": rating, "review_text": "Everything was delicious.", "order_id": order_id},
                prefix=self.category, rng=rng,
            ))
            cid4 = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid4, json.dumps({
                "review_id": f"REV-{rng.randint(10000, 99999)}", "rating": rating, "status": "published",
            })))
            msgs.append(assistant_msg(f"Thank you for the {rating}-star review! We're glad you enjoyed your meal. Hope to see you again!"))

            yield Example(messages=msgs)
            idx += 1
