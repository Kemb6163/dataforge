"""Knowledge base generator — KB search + follow-up SFT examples.

Produces ~100 examples covering:
- KB article search
- KB search + recommendation
- Multi-step: search → not found → suggest alternatives
- KB + ticket creation
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
from data_pools import KB_ARTICLES, get_customer_id, get_ticket_id


class KnowledgeBaseGenerator(SFTGenerator):

    @property
    def category(self) -> str:
        return "knowledge_base"

    @property
    def name(self) -> str:
        return "Knowledge Base"

    def expected_count(self) -> int:
        return 100

    def generate(self) -> Iterator[Example]:
        seed = self.config.get("seed", 42)
        error_cfg = self.config.get("error_injection", {})
        error_enabled = error_cfg.get("enabled", True)
        error_rate = error_cfg.get("base_rate", 0.10)
        custom_styles = self.config.get("styles", {})
        total = self.expected_count()
        idx = 0

        all_articles = []
        for cat, articles in KB_ARTICLES.items():
            for a in articles:
                all_articles.append({**a, "category": cat})

        # Pattern 1: Simple KB search (30 examples)
        search_queries = [
            "How do I set up SSO?", "What are the API rate limits?",
            "How to export my data?", "How do I reset my password?",
            "What payment methods do you accept?", "How to configure webhooks?",
            "How to upgrade my plan?", "Getting started guide",
            "How to connect Slack?", "Troubleshooting login issues",
            "How to use pagination?", "Database connector setup",
            "Understanding my invoice", "API authentication guide",
            "How to set up Jira integration?",
        ]
        for i in range(30):
            rng = make_rng(self.category, idx, seed)
            query = search_queries[i % len(search_queries)]
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg(query)]

            if error_enabled and (err_type := should_inject_error(self.category, idx, total, error_rate, seed)):
                msgs.append(tool_call_msg("search_knowledge_base", {"query": query}, prefix=self.category, rng=rng))
                cid = msgs[-1]["tool_calls"][0]["id"]
                msgs.append(tool_result_msg(cid, json.dumps(make_error_response(err_type))))
                msgs.append(assistant_msg(make_error_handling_response(err_type, rng)))
                yield Example(messages=msgs)
                idx += 1
                continue

            msgs.append(tool_call_msg("search_knowledge_base", {"query": query}, prefix=self.category, rng=rng))
            cid = msgs[-1]["tool_calls"][0]["id"]

            results = rng.sample(all_articles, min(rng.randint(1, 3), len(all_articles)))
            result_data = [{"id": a["id"], "title": a["title"], "summary": a["summary"]} for a in results]
            msgs.append(tool_result_msg(cid, json.dumps(result_data)))

            lines = [f"- **{a['title']}** ({a['id']}): {a['summary']}" for a in results]
            body = f"I found these relevant articles:\n" + "\n".join(lines)
            body += "\n\nWould you like me to look into any of these in more detail?"
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 2: Category-specific KB search (25 examples)
        categories = list(KB_ARTICLES.keys())
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            cat = rng.choice(categories)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            queries = [
                f"Show me {cat.replace('_', ' ')} articles",
                f"What's in the {cat.replace('_', ' ')} section?",
                f"I need help with {cat.replace('_', ' ')}",
            ]
            msgs = [user_msg(rng.choice(queries))]

            msgs.append(tool_call_msg(
                "search_knowledge_base",
                {"query": cat.replace("_", " "), "category": cat},
                prefix=self.category, rng=rng,
            ))
            cid = msgs[-1]["tool_calls"][0]["id"]

            results = KB_ARTICLES.get(cat, [])
            result_data = [{"id": a["id"], "title": a["title"], "summary": a["summary"]} for a in results]
            msgs.append(tool_result_msg(cid, json.dumps(result_data)))

            lines = [f"- **{a['title']}**: {a['summary']}" for a in results]
            body = f"Here are the {cat.replace('_', ' ')} articles:\n" + "\n".join(lines)
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 3: KB search → not found → suggest creating ticket (20 examples)
        for i in range(20):
            rng = make_rng(self.category, idx, seed)
            obscure_queries = [
                "How to configure LDAP with custom certificates?",
                "Can I use GraphQL instead of REST?",
                "How to set up multi-region failover?",
                "Custom SAML attribute mapping",
                "How to implement custom SSO with OIDC?",
                "Reverse proxy configuration for on-premise",
                "How to batch delete records via API?",
                "Custom webhook retry policy",
                "How to audit API key usage?",
                "Setting up IP allowlisting for API access",
            ]
            query = obscure_queries[i % len(obscure_queries)]
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            msgs = [user_msg(query)]
            msgs.append(tool_call_msg("search_knowledge_base", {"query": query}, prefix=self.category, rng=rng))
            cid = msgs[-1]["tool_calls"][0]["id"]
            msgs.append(tool_result_msg(cid, json.dumps([])))

            body = (
                "I wasn't able to find a knowledge base article specifically about that topic. "
                "Would you like me to:\n"
                "1. Create a support ticket so our team can help you directly\n"
                "2. Search with different keywords\n\n"
                "Our documentation is always growing, so it's possible this topic hasn't been covered yet."
            )
            msgs.append(assistant_msg(build_response(body, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1

        # Pattern 4: KB search → follow-up with ticket creation (25 examples)
        for i in range(25):
            rng = make_rng(self.category, idx, seed)
            query = rng.choice(search_queries)
            cust_id = get_customer_id(rng)
            style = pick_style(rng, custom_styles)
            structure = pick_structure(rng)

            # Turn 1: Search KB
            msgs = [user_msg(f"I'm having trouble with {query.lower()}. Customer: {cust_id}")]
            msgs.append(tool_call_msg("search_knowledge_base", {"query": query}, prefix=self.category, rng=rng))
            cid1 = msgs[-1]["tool_calls"][0]["id"]

            results = rng.sample(all_articles, min(2, len(all_articles)))
            result_data = [{"id": a["id"], "title": a["title"], "summary": a["summary"]} for a in results]
            msgs.append(tool_result_msg(cid1, json.dumps(result_data)))

            lines = [f"- **{a['title']}**: {a['summary']}" for a in results]
            body1 = f"Here are some articles that might help:\n" + "\n".join(lines)
            body1 += "\n\nDid any of these solve your issue, or should I create a support ticket?"
            msgs.append(assistant_msg(build_response(body1, style, structure, rng)))

            # Turn 2: User says KB didn't help → create ticket
            msgs.append(user_msg(rng.choice([
                "No, those didn't help. Please create a ticket.",
                "I already tried those. Can you open a ticket?",
                "None of those are relevant. I need human support.",
            ])))

            subject = f"Issue: {query.lower()}"
            msgs.append(tool_call_msg(
                "create_ticket",
                {"subject": subject, "description": f"KB articles not sufficient. Customer needs direct support for: {query}", "priority": "medium", "customer_id": cust_id, "category": "general"},
                prefix=self.category, rng=rng,
            ))
            cid2 = msgs[-1]["tool_calls"][0]["id"]
            ticket_id = get_ticket_id(rng)
            msgs.append(tool_result_msg(cid2, json.dumps({"ticket_id": ticket_id, "status": "open"})))

            body2 = (
                f"I've created ticket **{ticket_id}** for you. "
                "Our support team will follow up directly. "
                "Is there anything else I can help with?"
            )
            msgs.append(assistant_msg(build_response(body2, style, structure, rng)))

            yield Example(messages=msgs)
            idx += 1
