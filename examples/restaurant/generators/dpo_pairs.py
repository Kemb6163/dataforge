"""DPO preference pair generator for restaurant domain.

Produces ~60 DPO pairs covering:
- Good vs bad tool usage decisions
- Complete vs incomplete responses
- Correct vs incorrect tool arguments
- Polite vs dismissive tone
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import (
    user_msg, tool_call_msg, tool_result_msg, assistant_msg,
)
from dataforge.core.types import DPOPair
from dataforge.generation.base import DPOGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import (
    get_random_dish, get_random_dishes, get_customer_name,
    get_reservation_date, get_reservation_time, get_party_size,
    get_order_id, OUT_OF_SCOPE_REQUESTS,
)


class RestaurantDPOGenerator(DPOGenerator):
    """Generates DPO preference pairs for restaurant conversations."""

    @property
    def category(self) -> str:
        return "restaurant_dpo"

    @property
    def name(self) -> str:
        return "Restaurant DPO Pairs"

    def expected_count(self) -> int:
        return 60

    def generate(self) -> Iterator[DPOPair]:
        seed = self.config.get("seed", 42)
        idx = 0

        # Category 1: Tool usage — should use tool vs didn't (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            dish = get_random_dish(rng)
            query = rng.choice([
                f"What's in the {dish['name']}?",
                f"Tell me about the {dish['name']}.",
                f"How much is the {dish['name']}?",
            ])

            prompt = [user_msg(query)]

            # Chosen: uses tool correctly
            chosen_tool = tool_call_msg("get_dish_details", {"dish_id": dish["id"]}, prefix=self.category, rng=rng)
            chosen_result = tool_result_msg(
                chosen_tool["tool_calls"][0]["id"],
                json.dumps({"name": dish["name"], "price": dish["price"], "calories": dish.get("calories", 400)}),
            )
            chosen_response = assistant_msg(
                f"The **{dish['name']}** is ${dish['price']:.2f}. "
                f"It has about {dish.get('calories', 400)} calories. "
                "Would you like to know anything else about it?"
            )
            chosen = [chosen_tool, chosen_result, chosen_response]

            # Rejected: makes up information without using tool
            rejected = [assistant_msg(
                f"The {dish['name']} is a great dish! It's probably around $15-20 and has a moderate calorie count. "
                "I think it contains the usual ingredients for this type of dish."
            )]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1

        # Category 2: Complete vs incomplete responses (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            dishes = get_random_dishes(rng, 3, "mains")

            prompt = [user_msg("What main dishes do you recommend?")]

            # Chosen: detailed, structured response
            chosen_tool = tool_call_msg("search_menu", {"query": "recommended mains", "category": "mains"}, prefix=self.category, rng=rng)
            search_result = [{"name": d["name"], "id": d["id"], "price": d["price"], "dietary": d.get("dietary", [])} for d in dishes]
            chosen_result = tool_result_msg(
                chosen_tool["tool_calls"][0]["id"],
                json.dumps(search_result),
            )
            items = "\n".join(
                f"- **{d['name']}** — ${d['price']:.2f}" +
                (f" ({', '.join(d.get('dietary', []))})" if d.get('dietary') else "")
                for d in dishes
            )
            chosen_response = assistant_msg(
                f"Here are some of our recommended mains:\n{items}\n\n"
                "Would you like more details about any of these?"
            )
            chosen = [chosen_tool, chosen_result, chosen_response]

            # Rejected: vague, unhelpful response
            rejected_tool = tool_call_msg("search_menu", {"query": "mains"}, prefix=self.category, rng=rng)
            rejected_result = tool_result_msg(
                rejected_tool["tool_calls"][0]["id"],
                json.dumps(search_result),
            )
            rejected_response = assistant_msg(
                f"We have {dishes[0]['name']} and some other stuff."
            )
            rejected = [rejected_tool, rejected_result, rejected_response]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1

        # Category 3: Restraint — should NOT use tool for off-topic (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            request = OUT_OF_SCOPE_REQUESTS[i % len(OUT_OF_SCOPE_REQUESTS)]

            prompt = [user_msg(request)]

            # Chosen: polite refusal without tool use
            chosen = [assistant_msg(
                "I appreciate your question, but I'm specifically designed to help with "
                "restaurant-related tasks like searching our menu, making reservations, "
                "checking order status, and handling reviews. "
                "Is there anything restaurant-related I can assist you with?"
            )]

            # Rejected: tries to use a tool for something off-topic
            rejected_tool = tool_call_msg("search_menu", {"query": request}, prefix=self.category, rng=rng)
            rejected_result = tool_result_msg(
                rejected_tool["tool_calls"][0]["id"],
                json.dumps([]),
            )
            rejected_response = assistant_msg("I searched but couldn't find anything. Sorry!")
            rejected = [rejected_tool, rejected_result, rejected_response]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1

        # Category 4: Tone — professional vs dismissive (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            order_id = get_order_id(rng)

            prompt = [user_msg(f"Where's my order {order_id}? I've been waiting forever!")]

            result_data = {"order_id": order_id, "status": "cooking", "eta_minutes": 15}

            # Chosen: empathetic, helpful
            chosen_tool = tool_call_msg("get_order_status", {"order_id": order_id}, prefix=self.category, rng=rng)
            chosen_result = tool_result_msg(chosen_tool["tool_calls"][0]["id"], json.dumps(result_data))
            chosen_response = assistant_msg(
                f"I completely understand your frustration. Let me check on that right away.\n\n"
                f"Your order {order_id} is currently being cooked. The estimated time is about "
                f"15 more minutes. I know the wait can be frustrating — "
                "is there anything else I can help with in the meantime?"
            )
            chosen = [chosen_tool, chosen_result, chosen_response]

            # Rejected: dismissive tone
            rejected_tool = tool_call_msg("get_order_status", {"order_id": order_id}, prefix=self.category, rng=rng)
            rejected_result = tool_result_msg(rejected_tool["tool_calls"][0]["id"], json.dumps(result_data))
            rejected_response = assistant_msg(f"It says cooking. 15 minutes.")
            rejected = [rejected_tool, rejected_result, rejected_response]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1
