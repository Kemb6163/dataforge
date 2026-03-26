"""Ticket creation generator — multi-turn SFT examples.

Produces ~120 examples covering:
- Full intake: gather info → create ticket → confirm
- Quick creation (user provides all info upfront)
- Create + assign flow
- Create with customer lookup
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
    get_ticket_id, get_customer_id, get_customer_data, TICKET_SUBJECTS,
    TICKET_PRIORITIES,
)


class TicketCreationGenerator(SFTGenerator):

    @property
    def category(self) -> str:
        return "ticket_creation"

    @property
    def name(self) -> str:
        return "Ticket Creation"

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

        # Pattern 1: Full intake conversation (40 examples)
        for i in range(40):
            rng = make_rng(self.category, idx, seed)
            category = rng.choice(list(TICKET_SUBJECTS.keys()))
            subject = rng.choice(TICKET_SUBJECTS[category])
            cust = get_customer_data(rng)
            priority = rng.choice(TICKET_PRIORITIES)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            # Turn 1: User describes issue
            description_templates = [
                f"I'm having an issue: {subject.lower()}. My customer ID is {cust['customer_id']}.",
                f"Hi, I need help. {subject}. Account: {cust['customer_id']}.",
                f"There's a problem with my account ({cust['customer_id']}). {subject}.",
            ]
            msgs = [user_msg(rng.choice(description_templates))]

            # Assistant creates ticket
            detail = f"Customer reported: {subject.lower()}"
            msgs.append(tool_call_msg(
                "create_ticket",
                {
                    "subject": subject,
                    "description": detail,
                    "priority": priority,
                    "customer_id": cust["customer_id"],
                    "category": category,
                },
                prefix=self.category, rng=rng,
            ))
            cid = msgs[-1]["tool_calls"][0]["id"]

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs.append(tool_result_msg(cid, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            ticket_id = get_ticket_id(rng)
            result = {
                "ticket_id": ticket_id,
                "subject": subject,
                "status": "open",
                "priority": priority,
                "category": category,
            }
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            body = (
                f"I've created a support ticket for you:\n"
                f"- **Ticket ID**: {ticket_id}\n"
                f"- **Subject**: {subject}\n"
                f"- **Priority**: {priority}\n"
                f"- **Category**: {category}\n\n"
                f"Our team will review it shortly. Is there anything else you'd like to add?"
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Quick creation with all info (30 examples)
        for i in range(30):
            rng = make_rng(self.category, idx, seed)
            category = rng.choice(list(TICKET_SUBJECTS.keys()))
            subject = rng.choice(TICKET_SUBJECTS[category])
            cust_id = get_customer_id(rng)
            priority = rng.choice(TICKET_PRIORITIES)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            query = (
                f"Create a {priority} ticket for customer {cust_id}: "
                f"\"{subject}\". Category: {category}."
            )
            msgs = [user_msg(query)]

            detail = f"Customer {cust_id} reported: {subject}"
            msgs.append(tool_call_msg(
                "create_ticket",
                {
                    "subject": subject,
                    "description": detail,
                    "priority": priority,
                    "customer_id": cust_id,
                    "category": category,
                },
                prefix=self.category, rng=rng,
            ))
            cid = msgs[-1]["tool_calls"][0]["id"]

            ticket_id = get_ticket_id(rng)
            result = {"ticket_id": ticket_id, "status": "open", "priority": priority}
            msgs.append(tool_result_msg(cid, json.dumps(result)))

            body = f"Ticket **{ticket_id}** created — {priority} priority, category: {category}."
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: Create + immediately update/assign (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            category = rng.choice(list(TICKET_SUBJECTS.keys()))
            subject = rng.choice(TICKET_SUBJECTS[category])
            cust_id = get_customer_id(rng)
            priority = rng.choice(["high", "critical"])
            assignee = rng.choice(["Agent Smith", "Agent Lee", "Agent Patel"])
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg(
                f"Create a {priority} ticket for {cust_id}: {subject}. Assign to {assignee}."
            )]

            # Create ticket
            detail = f"Reported: {subject}"
            msgs.append(tool_call_msg(
                "create_ticket",
                {"subject": subject, "description": detail, "priority": priority, "customer_id": cust_id, "category": category},
                prefix=self.category, rng=rng,
            ))
            cid1 = msgs[-1]["tool_calls"][0]["id"]
            ticket_id = get_ticket_id(rng)
            msgs.append(tool_result_msg(cid1, json.dumps({"ticket_id": ticket_id, "status": "open"})))

            # Update with assignee
            msgs.append(tool_call_msg(
                "update_ticket",
                {"ticket_id": ticket_id, "assignee": assignee, "status": "in_progress"},
                prefix=self.category, rng=rng,
            ))
            cid2 = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid2, json.dumps({"ticket_id": ticket_id, "status": "in_progress", "assignee": assignee})))

            body = (
                f"Done! Ticket **{ticket_id}** has been created and assigned:\n"
                f"- Subject: {subject}\n"
                f"- Priority: {priority}\n"
                f"- Assignee: {assignee}\n"
                f"- Status: in_progress"
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: Lookup customer then create ticket (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            cust = get_customer_data(rng)
            category = rng.choice(list(TICKET_SUBJECTS.keys()))
            subject = rng.choice(TICKET_SUBJECTS[category])
            priority = rng.choice(TICKET_PRIORITIES)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg(
                f"I need to create a ticket. Customer email: {cust['email']}. Issue: {subject}"
            )]

            # Lookup customer
            msgs.append(tool_call_msg(
                "get_customer_info",
                {"email": cust["email"]},
                prefix=self.category, rng=rng,
            ))
            cid1 = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid1, json.dumps(cust)))

            # Create ticket with customer ID
            detail = f"Customer {cust['name']} ({cust['plan']} plan) reported: {subject}"
            msgs.append(tool_call_msg(
                "create_ticket",
                {"subject": subject, "description": detail, "priority": priority, "customer_id": cust["customer_id"], "category": category},
                prefix=self.category, rng=rng,
            ))
            cid2 = msgs[-1]["tool_calls"][0]["id"]
            ticket_id = get_ticket_id(rng)
            msgs.append(tool_result_msg(cid2, json.dumps({"ticket_id": ticket_id, "status": "open"})))

            body = (
                f"Found customer **{cust['name']}** ({cust['plan']} plan, {cust['tickets_count']} previous tickets).\n\n"
                f"Created ticket **{ticket_id}**:\n"
                f"- Subject: {subject}\n"
                f"- Priority: {priority}\n"
                f"- Category: {category}"
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1
