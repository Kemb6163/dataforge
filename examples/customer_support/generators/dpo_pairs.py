"""DPO preference pair generator for customer support domain.

Produces ~60 DPO pairs covering:
- Thorough vs lazy investigation
- Empathetic vs robotic tone
- Correct vs incorrect escalation decisions
- Tool use vs no tool use decisions
"""

from __future__ import annotations

import json
from typing import Iterator

from dataforge.core.rng import make_rng
from dataforge.core.messages import (
    user_msg, tool_call_msg, multi_tool_call_msg, tool_result_msg, assistant_msg,
)
from dataforge.core.types import DPOPair
from dataforge.generation.base import DPOGenerator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pools import (
    get_ticket_data, get_customer_data, get_ticket_id,
    ESCALATION_REASONS, OUT_OF_SCOPE_REQUESTS,
)


class SupportDPOGenerator(DPOGenerator):

    @property
    def category(self) -> str:
        return "support_dpo"

    @property
    def name(self) -> str:
        return "Support DPO Pairs"

    def expected_count(self) -> int:
        return 60

    def generate(self) -> Iterator[DPOPair]:
        seed = self.config.get("seed", 42)
        idx = 0

        # Category 1: Thorough investigation vs lazy response (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            cust = get_customer_data(rng)
            ticket = get_ticket_data(rng, customer_id=cust["customer_id"])

            prompt = [user_msg(f"Customer {cust['customer_id']} called about ticket {ticket['ticket_id']}. What's the situation?")]

            # Chosen: parallel lookup of customer + ticket
            chosen_calls = multi_tool_call_msg([
                ("get_customer_info", {"customer_id": cust["customer_id"]}),
                ("search_tickets", {"query": ticket["ticket_id"], "customer_id": cust["customer_id"]}),
            ], prefix=self.category, rng=rng)
            chosen_r1 = tool_result_msg(chosen_calls["tool_calls"][0]["id"], json.dumps(cust))
            chosen_r2 = tool_result_msg(chosen_calls["tool_calls"][1]["id"], json.dumps([{
                "ticket_id": ticket["ticket_id"], "subject": ticket["subject"],
                "status": ticket["status"], "priority": ticket["priority"],
            }]))
            chosen_resp = assistant_msg(
                f"Here's the full context:\n\n"
                f"**Customer:** {cust['name']} ({cust['plan']} plan, MRR ${cust['mrr']:.2f})\n"
                f"**Ticket:** {ticket['ticket_id']} — {ticket['subject']}\n"
                f"**Status:** {ticket['status']} | Priority: {ticket['priority']}\n\n"
                f"They've had {cust['tickets_count']} previous tickets. "
                "Would you like me to escalate or update the ticket?"
            )
            chosen = [chosen_calls, chosen_r1, chosen_r2, chosen_resp]

            # Rejected: no tool use, guessing
            rejected = [assistant_msg(
                f"Ticket {ticket['ticket_id']} is about some issue. "
                "I'd suggest checking the ticket system for more details."
            )]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1

        # Category 2: Empathetic vs robotic tone (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            cust = get_customer_data(rng)
            ticket = get_ticket_data(rng, customer_id=cust["customer_id"], category="technical")

            prompt = [user_msg(
                f"I've been having this issue for DAYS and nobody has helped me! "
                f"Ticket {ticket['ticket_id']}. I'm so frustrated!"
            )]

            # Both look up the ticket
            tool_msg = tool_call_msg("search_tickets", {"query": ticket["ticket_id"]}, prefix=self.category, rng=rng)
            result_msg = tool_result_msg(tool_msg["tool_calls"][0]["id"], json.dumps([{
                "ticket_id": ticket["ticket_id"], "subject": ticket["subject"],
                "status": "open", "priority": "medium",
            }]))

            # Chosen: empathetic, takes action
            chosen_resp = assistant_msg(
                f"I completely understand your frustration, and I sincerely apologize for the delay. "
                f"You're right that this has been open too long.\n\n"
                f"I can see ticket {ticket['ticket_id']} ({ticket['subject']}) is still open. "
                f"Let me escalate this to our senior team immediately so you get the attention you deserve. "
                "Would you also like me to update the priority to high?"
            )
            chosen = [tool_msg, result_msg, chosen_resp]

            # Rejected: dismissive, no empathy
            tool_msg2 = tool_call_msg("search_tickets", {"query": ticket["ticket_id"]}, prefix=self.category, rng=rng)
            result_msg2 = tool_result_msg(tool_msg2["tool_calls"][0]["id"], json.dumps([{
                "ticket_id": ticket["ticket_id"], "subject": ticket["subject"],
                "status": "open", "priority": "medium",
            }]))
            rejected_resp = assistant_msg(
                f"Ticket {ticket['ticket_id']} is open, medium priority. It will be handled in order."
            )
            rejected = [tool_msg2, result_msg2, rejected_resp]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1

        # Category 3: Correct escalation decision (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            cust = get_customer_data(rng)
            cust["plan"] = "enterprise"
            cust["mrr"] = round(rng.uniform(200, 500), 2)
            ticket = get_ticket_data(rng, customer_id=cust["customer_id"], category="technical")

            prompt = [user_msg(
                f"Customer {cust['customer_id']} says their entire system is down. Ticket: {ticket['ticket_id']}"
            )]

            # Chosen: checks customer value, escalates appropriately
            chosen_lookup = tool_call_msg("get_customer_info", {"customer_id": cust["customer_id"]}, prefix=self.category, rng=rng)
            chosen_r1 = tool_result_msg(chosen_lookup["tool_calls"][0]["id"], json.dumps(cust))
            chosen_escalate = tool_call_msg(
                "escalate_ticket",
                {"ticket_id": ticket["ticket_id"], "reason": f"Enterprise customer (MRR ${cust['mrr']:.2f}) reporting total system outage", "team": "engineering"},
                prefix=self.category, rng=rng,
            )
            chosen_r2 = tool_result_msg(chosen_escalate["tool_calls"][0]["id"], json.dumps({
                "ticket_id": ticket["ticket_id"], "escalated_to": "engineering", "new_priority": "critical",
            }))
            chosen_resp = assistant_msg(
                f"This is an enterprise customer (MRR ${cust['mrr']:.2f}). I've immediately escalated "
                f"ticket {ticket['ticket_id']} to the Engineering team with critical priority. "
                "A system-wide outage for an enterprise client requires immediate attention."
            )
            chosen = [chosen_lookup, chosen_r1, chosen_escalate, chosen_r2, chosen_resp]

            # Rejected: doesn't check customer, doesn't escalate
            rejected = [assistant_msg(
                f"I see the ticket. System issues can sometimes be resolved by restarting. "
                "Have they tried clearing their cache? I'll mark this as medium priority."
            )]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1

        # Category 4: Out-of-scope restraint (15 pairs)
        for i in range(15):
            rng = make_rng(self.category, idx, seed)
            request = OUT_OF_SCOPE_REQUESTS[i % len(OUT_OF_SCOPE_REQUESTS)]

            prompt = [user_msg(request)]

            # Chosen: polite refusal
            chosen = [assistant_msg(
                "I appreciate your request, but that's outside my capabilities as a support assistant. "
                "I can help you with ticket management, knowledge base searches, customer lookups, "
                "and escalations. Is there a support-related task I can assist with?"
            )]

            # Rejected: tries to use tools inappropriately
            rejected_tool = tool_call_msg("search_knowledge_base", {"query": request}, prefix=self.category, rng=rng)
            rejected_result = tool_result_msg(rejected_tool["tool_calls"][0]["id"], json.dumps([]))
            rejected_resp = assistant_msg("I searched our knowledge base but couldn't find anything about that. Sorry!")
            rejected = [rejected_tool, rejected_result, rejected_resp]

            yield DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
            idx += 1
