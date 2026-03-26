"""Analytics generator — parallel tool call SFT examples.

Produces ~120 examples covering:
- Multi-ticket status checks (parallel)
- Customer info + ticket search (parallel)
- Team workload analysis (multiple searches)
- Customer history deep dive
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
from data_pools import (
    get_ticket_data, get_customer_data, get_customer_id,
    TICKET_STATUSES, TICKET_PRIORITIES,
)


class AnalyticsGenerator(SFTGenerator):

    @property
    def category(self) -> str:
        return "analytics"

    @property
    def name(self) -> str:
        return "Analytics & Parallel"

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

        # Pattern 1: Customer info + their tickets (parallel) (40 examples)
        for i in range(40):
            rng = make_rng(self.category, idx, seed)
            cust = get_customer_data(rng)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Give me the full picture on customer {cust['customer_id']}",
                f"I need all info on {cust['customer_id']} — account and tickets",
                f"Pull up customer {cust['customer_id']}'s details and open tickets",
            ]
            msgs = [user_msg(rng.choice(queries))]

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs.append(tool_call_msg("get_customer_info", {"customer_id": cust["customer_id"]}, prefix=self.category, rng=rng))
                cid = msgs[-1]["tool_calls"][0]["id"]
                msgs.append(tool_result_msg(cid, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            # Parallel: get_customer_info + search_tickets
            calls = [
                ("get_customer_info", {"customer_id": cust["customer_id"]}),
                ("search_tickets", {"query": cust["customer_id"], "customer_id": cust["customer_id"]}),
            ]
            msgs.append(multi_tool_call_msg(calls, prefix=self.category, rng=rng))

            cid1 = msgs[-1]["tool_calls"][0]["id"]
            cid2 = msgs[-1]["tool_calls"][1]["id"]

            msgs.append(tool_result_msg(cid1, json.dumps(cust)))

            n_tickets = rng.randint(0, 4)
            tickets = [get_ticket_data(rng, customer_id=cust["customer_id"]) for _ in range(n_tickets)]
            ticket_results = [{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": t["status"], "priority": t["priority"]} for t in tickets]
            msgs.append(tool_result_msg(cid2, json.dumps(ticket_results)))

            body = (
                f"**Customer Profile: {cust['name']}**\n"
                f"- ID: {cust['customer_id']}\n"
                f"- Email: {cust['email']}\n"
                f"- Plan: {cust['plan']}\n"
                f"- Product: {cust['product']}\n"
                f"- MRR: ${cust['mrr']:.2f}\n"
                f"- Member since: {cust['created']}\n\n"
            )
            if tickets:
                body += f"**Open Tickets ({len(tickets)}):**\n"
                for t in tickets:
                    body += f"- {t['ticket_id']}: {t['subject']} [{t['status']}, {t['priority']}]\n"
            else:
                body += "**No open tickets.**"

            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Status dashboard (parallel searches) (35 examples)
        for i in range(35):
            rng = make_rng(self.category, idx, seed)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                "Give me a breakdown of open vs in-progress tickets",
                "What does our ticket queue look like right now?",
                "Show me the current ticket status breakdown",
                "How many tickets are in each status?",
            ]
            msgs = [user_msg(rng.choice(queries))]

            # Parallel searches for different statuses
            statuses = rng.sample(["open", "in_progress", "pending"], 2)
            calls = [
                ("search_tickets", {"query": s, "status": s})
                for s in statuses
            ]
            tc_msg = multi_tool_call_msg(calls, prefix=self.category, rng=rng)
            msgs.append(tc_msg)

            counts = {}
            for j, s in enumerate(statuses):
                cid = tc_msg["tool_calls"][j]["id"]
                n = rng.randint(2, 15)
                tickets = [get_ticket_data(rng) for _ in range(n)]
                for t in tickets:
                    t["status"] = s
                result = [{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": s, "priority": t["priority"]} for t in tickets]
                msgs.append(tool_result_msg(cid, json.dumps(result)))
                counts[s] = n

            body = "**Ticket Queue Summary:**\n"
            for s, count in counts.items():
                body += f"- {s.replace('_', ' ').title()}: {count} tickets\n"
            total_count = sum(counts.values())
            body += f"\n**Total active:** {total_count} tickets"

            high_count = rng.randint(0, 3)
            if high_count > 0:
                body += f"\n\nNote: {high_count} of these are high/critical priority and may need immediate attention."

            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Priority comparison (parallel) (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg("How many critical and high priority tickets do we have?")]

            calls = [
                ("search_tickets", {"query": "critical", "priority": "critical"}),
                ("search_tickets", {"query": "high", "priority": "high"}),
            ]
            msgs.append(multi_tool_call_msg(calls, prefix=self.category, rng=rng))

            crit_n = rng.randint(0, 5)
            high_n = rng.randint(2, 10)

            crit_tickets = [get_ticket_data(rng) for _ in range(crit_n)]
            for t in crit_tickets:
                t["priority"] = "critical"
            high_tickets = [get_ticket_data(rng) for _ in range(high_n)]
            for t in high_tickets:
                t["priority"] = "high"

            cid1 = msgs[-1]["tool_calls"][0]["id"]
            cid2 = msgs[-1]["tool_calls"][1]["id"]
            msgs.append(tool_result_msg(cid1, json.dumps([{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": t["status"]} for t in crit_tickets])))
            msgs.append(tool_result_msg(cid2, json.dumps([{"ticket_id": t["ticket_id"], "subject": t["subject"], "status": t["status"]} for t in high_tickets])))

            body = f"**Priority Overview:**\n"
            body += f"- Critical: {crit_n} ticket(s)\n"
            body += f"- High: {high_n} ticket(s)\n"
            body += f"- **Total urgent:** {crit_n + high_n}\n"

            if crit_n > 0:
                body += "\n**Critical tickets:**\n"
                for t in crit_tickets:
                    body += f"- {t['ticket_id']}: {t['subject']}\n"

            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Customer history + KB search (parallel) (20 examples)
        for i in range(20):
            rng = make_rng(self.category, idx, seed)
            cust = get_customer_data(rng)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            issue = rng.choice(["API errors", "slow performance", "login issues", "billing discrepancy"])
            msgs = [user_msg(f"Customer {cust['customer_id']} is reporting {issue}. Get their info and check if we have KB articles.")]

            calls = [
                ("get_customer_info", {"customer_id": cust["customer_id"]}),
                ("search_knowledge_base", {"query": issue}),
            ]
            msgs.append(multi_tool_call_msg(calls, prefix=self.category, rng=rng))

            cid1 = msgs[-1]["tool_calls"][0]["id"]
            cid2 = msgs[-1]["tool_calls"][1]["id"]

            msgs.append(tool_result_msg(cid1, json.dumps(cust)))

            kb_results = [
                {"id": f"KB-{rng.randint(100, 999)}", "title": f"Troubleshooting {issue.title()}", "summary": f"Steps to resolve common {issue} problems."},
            ]
            msgs.append(tool_result_msg(cid2, json.dumps(kb_results)))

            body = (
                f"**Customer:** {cust['name']} ({cust['plan']} plan)\n"
                f"**Issue:** {issue}\n\n"
                f"**Relevant KB article:** {kb_results[0]['title']} ({kb_results[0]['id']})\n"
                f"Summary: {kb_results[0]['summary']}\n\n"
                "I'd suggest sharing this article with the customer first. "
                "If it doesn't resolve the issue, we can create a ticket."
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1
