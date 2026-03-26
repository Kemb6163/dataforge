"""Escalation generator — complex decision + no-tool restraint examples.

Produces ~100 examples covering:
- When to escalate vs resolve at L1
- Escalation with context
- No-tool restraint for out-of-scope requests
- Multi-step: investigate → decide → escalate
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import user_msg, tool_call_msg, tool_result_msg, assistant_msg
from dataforge.core.styles import pick_style, pick_structure, build_response
from dataforge.core.types import Example
from dataforge.generation.base import SFTGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import (
    get_ticket_id, get_ticket_data, get_customer_data,
    ESCALATION_REASONS, TEAMS, OUT_OF_SCOPE_REQUESTS,
)


class EscalationGenerator(SFTGenerator):

    @property
    def category(self) -> str:
        return "escalation"

    @property
    def name(self) -> str:
        return "Escalation & Restraint"

    def expected_count(self) -> int:
        return 110

    def generate(self) -> Iterator[Example]:
        seed = self.config.get("seed", 42)
        custom_styles = self.config.get("styles", {})
        idx = 0

        # Pattern 1: Direct escalation request (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            ticket = get_ticket_data(rng, category="technical")
            team = rng.choice(TEAMS)
            reason = rng.choice(ESCALATION_REASONS)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Escalate ticket {ticket['ticket_id']} to {team}. Reason: {reason}",
                f"I need to escalate {ticket['ticket_id']} — {reason}",
                f"Please send {ticket['ticket_id']} to the {team} team. {reason}",
            ]

            msgs = [user_msg(rng.choice(queries))]
            msgs.append(tool_call_msg(
                "escalate_ticket",
                {"ticket_id": ticket["ticket_id"], "reason": reason, "team": team},
                prefix=self.category, rng=rng,
            ))
            cid = msgs[-1]["tool_calls"][0]["id"]

            result = {
                "ticket_id": ticket["ticket_id"],
                "escalated_to": team,
                "new_priority": "high",
                "estimated_response": "2 hours",
            }
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            body = (
                f"Ticket **{ticket['ticket_id']}** has been escalated to the {team.replace('_', ' ').title()} team.\n"
                f"- Reason: {reason}\n"
                f"- New priority: high\n"
                f"- Estimated response: ~2 hours\n\n"
                "The team will review and follow up directly with the customer."
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Investigate → Decide → Escalate (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            cust = get_customer_data(rng)
            ticket = get_ticket_data(rng, customer_id=cust["customer_id"], category="technical")
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            # Turn 1: Ask about a ticket
            msgs = [user_msg(f"Ticket {ticket['ticket_id']} has been open for days. The customer is upset. What should we do?")]

            # Look up customer info
            msgs.append(tool_call_msg("get_customer_info", {"customer_id": cust["customer_id"]}, prefix=self.category, rng=rng))
            cid1 = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid1, json.dumps(cust)))

            is_high_value = cust["plan"] in ("professional", "enterprise") or cust["mrr"] > 100
            if is_high_value:
                body1 = (
                    f"This is a **{cust['plan']}** customer (MRR: ${cust['mrr']:.2f}, {cust['tickets_count']} previous tickets). "
                    f"Given the account value and the time this ticket has been open, I recommend escalating to the senior support team immediately."
                )
            else:
                body1 = (
                    f"Customer is on the {cust['plan']} plan with {cust['tickets_count']} previous tickets. "
                    "Let me check if there's a pattern in their tickets before deciding on escalation."
                )
            msgs.append(assistant_msg(build_response(body1, style, structure, rng)))

            # Turn 2: Escalate
            msgs.append(user_msg("Go ahead and escalate it."))

            team = "senior_support" if is_high_value else "engineering"
            reason = (
                f"High-value {cust['plan']} customer (MRR ${cust['mrr']:.2f}). Ticket open for extended period."
                if is_high_value else
                f"Ticket open for extended period. Standard troubleshooting exhausted."
            )

            msgs.append(tool_call_msg(
                "escalate_ticket",
                {"ticket_id": ticket["ticket_id"], "reason": reason, "team": team},
                prefix=self.category, rng=rng,
            ))
            cid2 = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid2, json.dumps({
                "ticket_id": ticket["ticket_id"], "escalated_to": team, "new_priority": "high",
            })))

            body2 = (
                f"Escalated to {team.replace('_', ' ').title()}. "
                f"The customer ({cust['name']}) will be contacted within 2 hours."
            )
            msgs.append(assistant_msg(build_response(body2, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Resolve without escalation (10 examples)
        for i in range(10):
            rng = make_rng(self.category, idx, seed)
            ticket = get_ticket_data(rng, category=rng.choice(["general", "account"]))
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg(f"Should we escalate ticket {ticket['ticket_id']}? It's a {ticket['category']} issue.")]

            # Look up ticket context
            msgs.append(tool_call_msg("search_tickets", {"query": ticket["ticket_id"]}, prefix=self.category, rng=rng))
            cid = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid, json.dumps([{
                "ticket_id": ticket["ticket_id"],
                "subject": ticket["subject"],
                "status": ticket["status"],
                "priority": "low",
                "category": ticket["category"],
            }])))

            body = (
                f"Based on my review, ticket {ticket['ticket_id']} ({ticket['subject']}) "
                f"is a standard {ticket['category']} issue with low priority. "
                "I don't think escalation is necessary — this can be resolved at L1. "
                "Would you like me to update the ticket with a resolution, or would you prefer to escalate anyway?"
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Out-of-scope requests — no tool calls (50 examples)
        for i in range(50):
            rng = make_rng(self.category, idx, seed)
            request = OUT_OF_SCOPE_REQUESTS[i % len(OUT_OF_SCOPE_REQUESTS)]
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            refusals = [
                "I appreciate your request, but that falls outside what I can help with. "
                "I'm designed to assist with support ticket management, knowledge base searches, "
                "customer information lookups, and ticket escalations. "
                "Is there a support-related task I can help you with?",

                "That's not something I'm able to do. My capabilities are focused on "
                "customer support operations — managing tickets, searching our knowledge base, "
                "looking up customer info, and handling escalations. "
                "How can I assist you with a support task?",

                "I'm sorry, but I can't help with that request. I'm a customer support "
                "assistant with tools for ticket management, KB search, and customer lookup. "
                "Would you like help with any of those?",

                "That's outside my scope. I specialize in support operations: "
                "ticket search/creation/escalation, knowledge base queries, and customer info. "
                "Let me know if you need help with any of those!",
            ]

            msgs = [user_msg(request)]
            msgs.append(assistant_msg(rng.choice(refusals)))

            yield Example(messages=msgs)
            idx += 1
