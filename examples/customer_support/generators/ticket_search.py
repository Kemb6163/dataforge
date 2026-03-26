"""Ticket search generator — single tool call SFT examples.

Produces ~120 examples covering:
- Search by keyword
- Search by status/priority filter
- Search by customer
- Empty result handling
- Error handling
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
    get_ticket_data, get_customer_id, TICKET_STATUSES, TICKET_PRIORITIES,
    TICKET_SUBJECTS,
)


class TicketSearchGenerator(SFTGenerator):

    @property
    def category(self) -> str:
        return "ticket_search"

    @property
    def name(self) -> str:
        return "Ticket Search"

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
        keywords = [
            "login issue", "billing problem", "slow performance", "API error",
            "account locked", "export failed", "webhook", "SSO",
            "payment", "integration", "permission denied", "timeout",
            "data missing", "upgrade", "downgrade",
        ]
        for i in range(30):
            rng = make_rng(self.category, idx, seed)
            keyword = rng.choice(keywords)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Search for tickets about {keyword}",
                f"Find any open tickets related to {keyword}",
                f"Are there any tickets mentioning {keyword}?",
                f"Look up tickets: {keyword}",
            ]

            msgs = [user_msg(rng.choice(queries))]

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs.append(tool_call_msg("search_tickets", {"query": keyword}, prefix=self.category, rng=rng))
                cid = msgs[-1]["tool_calls"][0]["id"]
                msgs.append(tool_result_msg(cid, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            msgs.append(tool_call_msg("search_tickets", {"query": keyword}, prefix=self.category, rng=rng))
            cid = msgs[-1]["tool_calls"][0]["id"]

            n_results = rng.randint(0, 5)
            tickets = [get_ticket_data(rng) for _ in range(n_results)]
            result = [{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": t["status"], "priority": t["priority"]} for t in tickets]
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            if tickets:
                lines = []
                for t in tickets:
                    lines.append(f"- **{t['ticket_id']}**: {t['subject']} [{t['status']}] ({t['priority']})")
                body = f"Found {len(tickets)} ticket(s) related to \"{keyword}\":\n" + "\n".join(lines)
            else:
                body = ""
            msgs.append(assistant_msg(build_response(body, style, structure, rng, no_result=(not tickets))))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Status filter search (30 examples)
        for i in range(30):
            rng = make_rng(self.category, idx, seed)
            status = rng.choice(TICKET_STATUSES)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Show me all {status} tickets",
                f"What tickets are currently {status}?",
                f"List {status} tickets",
                f"How many tickets are in {status} status?",
            ]

            msgs = [user_msg(rng.choice(queries))]
            msgs.append(tool_call_msg("search_tickets", {"query": status, "status": status}, prefix=self.category, rng=rng))
            cid = msgs[-1]["tool_calls"][0]["id"]

            n_results = rng.randint(1, 6)
            tickets = [get_ticket_data(rng) for _ in range(n_results)]
            for t in tickets:
                t["status"] = status
            result = [{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": t["status"], "priority": t["priority"]} for t in tickets]
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            lines = [f"- **{t['ticket_id']}**: {t['subject']} ({t['priority']})" for t in tickets]
            body = f"There are {len(tickets)} {status} ticket(s):\n" + "\n".join(lines)
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Priority filter (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            priority = rng.choice(TICKET_PRIORITIES)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Show me all {priority} priority tickets",
                f"What {priority} tickets do we have?",
                f"List tickets with {priority} priority",
            ]

            msgs = [user_msg(rng.choice(queries))]
            msgs.append(tool_call_msg("search_tickets", {"query": priority, "priority": priority}, prefix=self.category, rng=rng))
            cid = msgs[-1]["tool_calls"][0]["id"]

            n_results = rng.randint(1, 5)
            tickets = [get_ticket_data(rng) for _ in range(n_results)]
            for t in tickets:
                t["priority"] = priority
            result = [{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": t["status"], "priority": priority} for t in tickets]
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            lines = [f"- **{t['ticket_id']}**: {t['subject']} [{t['status']}]" for t in tickets]
            body = f"{len(tickets)} {priority}-priority ticket(s):\n" + "\n".join(lines)
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Customer-specific search (20 examples)
        for i in range(20):
            rng = make_rng(self.category, idx, seed)
            cust_id = get_customer_id(rng)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Show tickets for customer {cust_id}",
                f"What issues has customer {cust_id} reported?",
                f"Pull up all tickets for {cust_id}",
            ]

            msgs = [user_msg(rng.choice(queries))]
            msgs.append(tool_call_msg("search_tickets", {"query": cust_id, "customer_id": cust_id}, prefix=self.category, rng=rng))
            cid = msgs[-1]["tool_calls"][0]["id"]

            n_results = rng.randint(0, 4)
            tickets = [get_ticket_data(rng, customer_id=cust_id) for _ in range(n_results)]
            result = [{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": t["status"], "priority": t["priority"]} for t in tickets]
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            if tickets:
                lines = [f"- **{t['ticket_id']}**: {t['subject']} [{t['status']}, {t['priority']}]" for t in tickets]
                body = f"Customer {cust_id} has {len(tickets)} ticket(s):\n" + "\n".join(lines)
            else:
                body = ""
            msgs.append(assistant_msg(build_response(body, style, structure, rng, no_result=(not tickets))))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 5: Combined filter (15 examples)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            status = rng.choice(["open", "in_progress"])
            priority = rng.choice(["high", "critical"])
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Find all {status} {priority} tickets",
                f"Show {priority}-priority tickets that are {status}",
                f"I need to see {status} tickets with {priority} priority",
            ]

            msgs = [user_msg(rng.choice(queries))]
            msgs.append(tool_call_msg(
                "search_tickets",
                {"query": f"{status} {priority}", "status": status, "priority": priority},
                prefix=self.category, rng=rng,
            ))
            cid = msgs[-1]["tool_calls"][0]["id"]

            n_results = rng.randint(1, 4)
            tickets = [get_ticket_data(rng) for _ in range(n_results)]
            for t in tickets:
                t["status"] = status
                t["priority"] = priority
            result = [{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": status, "priority": priority, "customer_id": t["customer_id"]} for t in tickets]
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            lines = [f"- **{t['ticket_id']}**: {t['subject']} (Customer: {t['customer_id']})" for t in tickets]
            body = f"{len(tickets)} {status} {priority}-priority ticket(s):\n" + "\n".join(lines)
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1
