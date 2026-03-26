"""Order management generator — parallel tool call SFT examples.

Produces ~120 examples covering:
- Single order status check
- Multiple order status checks (parallel tool calls)
- Order tracking with follow-up questions
- Error handling (order not found)
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import (
    user_msg, tool_call_msg, multi_tool_call_msg, tool_result_msg, assistant_msg,
)
from dataforge.core.styles import pick_style, pick_structure, build_response
from dataforge.core.errors import should_inject_error, make_error_response, make_error_handling_response
from dataforge.core.types import Example
from dataforge.generation.base import SFTGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import get_order_id, ORDER_STATUSES, get_random_dish


class OrderManagementGenerator(SFTGenerator):
    """Generates order tracking examples including parallel tool calls."""

    @property
    def category(self) -> str:
        return "order_management"

    @property
    def name(self) -> str:
        return "Order Management"

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

        # Pattern 1: Single order check (35 examples)
        for i in range(35):
            rng = make_rng(self.category, idx, seed)
            order_id = get_order_id(rng)
            status = rng.choice(ORDER_STATUSES)

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"What's the status of order {order_id}?",
                f"Can you check on my order {order_id}?",
                f"Where's my order? The number is {order_id}.",
                f"I placed order {order_id}, how long until it's ready?",
                f"Track order {order_id} please.",
            ]

            msgs = [user_msg(rng.choice(queries))]

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs.append(tool_call_msg("get_order_status", {"order_id": order_id}, prefix=self.category, rng=rng))
                call_id = msgs[-1]["tool_calls"][0]["id"]
                msgs.append(tool_result_msg(call_id, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            msgs.append(tool_call_msg("get_order_status", {"order_id": order_id}, prefix=self.category, rng=rng))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            result = {
                "order_id": order_id,
                "status": status["status"],
                "eta_minutes": status["eta_minutes"],
                "items": [get_random_dish(rng)["name"] for _ in range(rng.randint(1, 3))],
            }
            msgs.append(tool_result_msg(call_id, json.dumps(result)))

            if status["eta_minutes"] > 0:
                body = (
                    f"Your order {order_id} is currently **{status['status']}**. "
                    f"{status['message']} Estimated time: about {status['eta_minutes']} minutes."
                )
            else:
                body = f"Your order {order_id}: {status['message']}"

            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Multiple orders — parallel tool calls (50 examples)
        for i in range(50):
            rng = make_rng(self.category, idx, seed)
            num_orders = rng.randint(2, 3)
            order_ids = [get_order_id(rng) for _ in range(num_orders)]

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            ids_str = ", ".join(order_ids)
            queries = [
                f"Can you check on orders {ids_str}?",
                f"I have multiple orders: {ids_str}. What's their status?",
                f"Status update on {ids_str} please.",
                f"Where are my orders {ids_str}?",
            ]

            msgs = [user_msg(rng.choice(queries))]

            # Parallel tool calls
            calls = [("get_order_status", {"order_id": oid}) for oid in order_ids]
            tc_msg = multi_tool_call_msg(calls, prefix=self.category, rng=rng)
            msgs.append(tc_msg)

            # Tool results for each
            statuses = []
            for j, oid in enumerate(order_ids):
                call_id = tc_msg["tool_calls"][j]["id"]
                status = rng.choice(ORDER_STATUSES)
                result = {
                    "order_id": oid,
                    "status": status["status"],
                    "eta_minutes": status["eta_minutes"],
                    "items": [get_random_dish(rng)["name"]],
                }
                msgs.append(tool_result_msg(call_id, json.dumps(result)))
                statuses.append((oid, status))

            # Build combined response
            parts = []
            for oid, status in statuses:
                if status["eta_minutes"] > 0:
                    parts.append(f"**{oid}**: {status['status']} — ~{status['eta_minutes']} min remaining")
                else:
                    parts.append(f"**{oid}**: {status['status']}")

            body = "Here's the status of your orders:\n" + "\n".join(f"- {p}" for p in parts)
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Order check + follow-up (20 examples)
        for i in range(20):
            rng = make_rng(self.category, idx, seed)
            order_id = get_order_id(rng)
            status = rng.choice([s for s in ORDER_STATUSES if s["status"] not in ("served", "cancelled")])

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg(f"Check order {order_id} please.")]
            msgs.append(tool_call_msg("get_order_status", {"order_id": order_id}, prefix=self.category, rng=rng))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            items = [get_random_dish(rng)["name"] for _ in range(rng.randint(2, 4))]
            result = {
                "order_id": order_id,
                "status": status["status"],
                "eta_minutes": status["eta_minutes"],
                "items": items,
            }
            msgs.append(tool_result_msg(call_id, json.dumps(result)))

            body = (
                f"Order {order_id} is **{status['status']}**. "
                f"ETA: ~{status['eta_minutes']} minutes.\n"
                f"Items: {', '.join(items)}"
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            # Follow-up
            followups = [
                "Can you check again in a few minutes?",
                "That's taking a while. Is everything okay?",
                "Thanks! What about my other order?",
            ]
            msgs.append(user_msg(rng.choice(followups)))

            followup_responses = [
                "I'd recommend checking back in about 5 minutes for an update. The kitchen is working on it!",
                "The kitchen is busy but your order is in the queue. It should be ready within the estimated time.",
                "I understand the wait can be frustrating. Rest assured your order is being prepared. Would you like me to check again shortly?",
            ]
            msgs.append(assistant_msg(rng.choice(followup_responses)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Order not found (15 examples)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            order_id = f"ORD-{rng.randint(10000, 99999)}"

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg(f"What's happening with order {order_id}?")]
            msgs.append(tool_call_msg("get_order_status", {"order_id": order_id}, prefix=self.category, rng=rng))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            not_found = {"error": True, "code": "NOT_FOUND", "message": f"Order {order_id} not found"}
            msgs.append(tool_result_msg(call_id, json.dumps(not_found)))

            body = (
                f"I couldn't find an order with ID {order_id}. "
                "Could you double-check the order number? It should be in the format ORD-XXXXX. "
                "You can find it in your confirmation email or receipt."
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1
