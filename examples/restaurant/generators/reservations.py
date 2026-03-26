"""Reservation generator — multi-turn SFT examples.

Produces ~120 examples covering:
- Check availability then book
- Direct booking attempts
- Full multi-turn: ask availability → unavailable → suggest alternative → book
- Special requests handling
- Error handling (fully booked, invalid dates)
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import (
    user_msg, tool_call_msg, tool_result_msg, assistant_msg,
)
from dataforge.core.styles import pick_style, pick_structure, build_response
from dataforge.core.errors import should_inject_error, make_error_response, make_error_handling_response
from dataforge.core.types import Example
from dataforge.generation.base import SFTGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import (
    get_customer_name, get_reservation_date, get_reservation_time,
    get_party_size, SPECIAL_REQUESTS,
)


class ReservationGenerator(SFTGenerator):
    """Generates multi-turn reservation conversation examples."""

    @property
    def category(self) -> str:
        return "reservations"

    @property
    def name(self) -> str:
        return "Reservations"

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

        # Pattern 1: Check availability → Book (40 examples)
        for i in range(40):
            rng = make_rng(self.category, idx, seed)
            cust_name = get_customer_name(rng)
            date = get_reservation_date(rng)
            time = get_reservation_time(rng)
            party = get_party_size(rng)
            special = rng.choice(SPECIAL_REQUESTS)

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            # User asks to check availability
            user_queries = [
                f"I'd like a table for {party} on {date} at {time}. Is that available?",
                f"Can I book a table for {party} people on {date} at {time}?",
                f"Do you have availability for {party} on {date} around {time}?",
                f"We're {party} people, can we come on {date} at {time}?",
                f"Check if you have a table for {party} on {date}, {time} please.",
            ]
            query1 = rng.choice(user_queries)

            msgs = [user_msg(query1)]

            # Tool call: check_availability
            msgs.append(tool_call_msg(
                "check_availability",
                {"date": date, "time": time, "party_size": party},
                prefix=self.category, rng=rng,
            ))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs.append(tool_result_msg(call_id, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            avail_result = {"available": True, "date": date, "time": time, "party_size": party}
            msgs.append(tool_result_msg(call_id, json.dumps(avail_result)))

            confirm_body = f"A table for {party} is available on {date} at {time}. Would you like me to book it?"
            msgs.append(assistant_msg(build_response(confirm_body, style, structure, rng)))

            # User confirms
            confirm_queries = [
                f"Yes, please book it under {cust_name}.",
                f"Great, book it for {cust_name}.",
                f"Perfect! Name is {cust_name}." + (f" {special}" if special else ""),
                f"Yes please. The name is {cust_name}.",
            ]
            msgs.append(user_msg(rng.choice(confirm_queries)))

            # Tool call: make_reservation
            res_args = {"date": date, "time": time, "party_size": party, "name": cust_name}
            if special:
                res_args["special_requests"] = special
            msgs.append(tool_call_msg("make_reservation", res_args, prefix=self.category, rng=rng))
            call_id2 = msgs[-1]["tool_calls"][0]["id"]

            booking_result = {
                "confirmation_id": f"RES-{rng.randint(10000, 99999)}",
                "date": date,
                "time": time,
                "party_size": party,
                "name": cust_name,
            }
            msgs.append(tool_result_msg(call_id2, json.dumps(booking_result)))

            book_body = (
                f"Your reservation is confirmed! Here are the details:\n"
                f"- Confirmation: {booking_result['confirmation_id']}\n"
                f"- Date: {date} at {time}\n"
                f"- Party size: {party}\n"
                f"- Name: {cust_name}"
            )
            if special:
                book_body += f"\n- Special request: {special}"
            msgs.append(assistant_msg(build_response(book_body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Unavailable → Suggest alternative → Book (40 examples)
        for i in range(40):
            rng = make_rng(self.category, idx, seed)
            cust_name = get_customer_name(rng)
            date = get_reservation_date(rng)
            time = get_reservation_time(rng)
            party = get_party_size(rng)

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            query1 = rng.choice([
                f"Table for {party} on {date} at {time}?",
                f"I want to reserve a table for {party} on {date}, {time}.",
                f"Is there space for {party} on {date} at {time}?",
            ])

            msgs = [user_msg(query1)]

            msgs.append(tool_call_msg(
                "check_availability",
                {"date": date, "time": time, "party_size": party},
                prefix=self.category, rng=rng,
            ))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            # Suggest alternative time
            alt_hour = int(time.split(":")[0]) + rng.choice([-1, 1, 2])
            alt_hour = max(11, min(21, alt_hour))
            alt_time = f"{alt_hour:02d}:{rng.choice(['00', '15', '30', '45'])}"

            unavail_result = {
                "available": False,
                "date": date,
                "time": time,
                "alternatives": [{"time": alt_time, "available": True}],
            }
            msgs.append(tool_result_msg(call_id, json.dumps(unavail_result)))

            alt_body = (
                f"Unfortunately, we don't have a table for {party} at {time} on {date}. "
                f"However, I can see that {alt_time} is available. Would that work for you?"
            )
            msgs.append(assistant_msg(build_response(alt_body, style, structure, rng)))

            # User accepts alternative
            accept_queries = [
                f"Sure, {alt_time} works. Book it for {cust_name}.",
                f"That's fine, go ahead and book {alt_time} under {cust_name}.",
                f"OK let's do {alt_time}. Name: {cust_name}.",
            ]
            msgs.append(user_msg(rng.choice(accept_queries)))

            msgs.append(tool_call_msg(
                "make_reservation",
                {"date": date, "time": alt_time, "party_size": party, "name": cust_name},
                prefix=self.category, rng=rng,
            ))
            call_id2 = msgs[-1]["tool_calls"][0]["id"]

            booking_result = {
                "confirmation_id": f"RES-{rng.randint(10000, 99999)}",
                "date": date,
                "time": alt_time,
                "party_size": party,
                "name": cust_name,
            }
            msgs.append(tool_result_msg(call_id2, json.dumps(booking_result)))

            book_body = (
                f"All set! Your reservation is confirmed:\n"
                f"- Confirmation: {booking_result['confirmation_id']}\n"
                f"- Date: {date} at {alt_time}\n"
                f"- Party: {party} guests\n"
                f"- Name: {cust_name}"
            )
            msgs.append(assistant_msg(build_response(book_body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Direct booking (no availability check) (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            cust_name = get_customer_name(rng)
            date = get_reservation_date(rng)
            time = get_reservation_time(rng)
            party = get_party_size(rng)
            special = rng.choice(SPECIAL_REQUESTS)

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            query = rng.choice([
                f"Book a table for {party} on {date} at {time} under {cust_name}.",
                f"I need a reservation: {party} people, {date}, {time}. Name is {cust_name}.",
                f"Please reserve for {cust_name}, party of {party}, {date} at {time}.",
            ])
            if special:
                query += f" {special}"

            msgs = [user_msg(query)]

            res_args = {"date": date, "time": time, "party_size": party, "name": cust_name}
            if special:
                res_args["special_requests"] = special

            msgs.append(tool_call_msg("make_reservation", res_args, prefix=self.category, rng=rng))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            booking_result = {
                "confirmation_id": f"RES-{rng.randint(10000, 99999)}",
                "date": date,
                "time": time,
                "party_size": party,
                "name": cust_name,
            }
            msgs.append(tool_result_msg(call_id, json.dumps(booking_result)))

            book_body = (
                f"Your reservation has been booked!\n"
                f"- Confirmation: {booking_result['confirmation_id']}\n"
                f"- Date: {date} at {time}\n"
                f"- Party: {party}\n"
                f"- Name: {cust_name}"
            )
            msgs.append(assistant_msg(build_response(book_body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Availability check only (15 examples)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            date = get_reservation_date(rng)
            time = get_reservation_time(rng)
            party = get_party_size(rng)
            available = rng.random() > 0.3

            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            query = rng.choice([
                f"Just checking — do you have space for {party} on {date} at {time}?",
                f"Are you open on {date}? I'd need a table for {party} at {time}.",
                f"Quick question: any tables for {party} at {time} on {date}?",
            ])

            msgs = [user_msg(query)]

            msgs.append(tool_call_msg(
                "check_availability",
                {"date": date, "time": time, "party_size": party},
                prefix=self.category, rng=rng,
            ))
            call_id = msgs[-1]["tool_calls"][0]["id"]

            result = {"available": available, "date": date, "time": time, "party_size": party}
            msgs.append(tool_result_msg(call_id, json.dumps(result)))

            if available:
                body = f"Yes, we have a table available for {party} on {date} at {time}. Would you like me to book it?"
            else:
                body = f"I'm sorry, but we're fully booked for {party} guests at {time} on {date}. Would you like me to check a different time?"
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1
