"""Customer support data pools for generating realistic examples."""

from __future__ import annotations

from dataforge.core.rng import make_rng
from dataforge.generation.pools import fake_name, fake_id, fake_email, fake_date


CUSTOMER_FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Quinn", "Avery",
    "Jamie", "Drew", "Cameron", "Blake", "Sage", "Reese", "Dakota", "Hayden",
    "Skyler", "Finley", "Rowan", "Emerson", "Harper", "Logan", "Spencer", "Kai",
]

CUSTOMER_LAST_NAMES = [
    "Chen", "Patel", "Kim", "Nguyen", "Muller", "Santos", "Johansson", "Singh",
    "Tanaka", "Rossi", "Andersen", "Kowalski", "Fernandez", "Murphy", "Wang",
    "Okafor", "Al-Rashidi", "Cohen", "Petrov", "Nakamura", "Dubois", "Ibrahim",
]

PRODUCT_NAMES = [
    "CloudSync Pro", "DataVault Enterprise", "TeamFlow", "AnalyticsPro",
    "SecureAuth", "APIGateway", "DeployBot", "MonitorHub",
]

SUBSCRIPTION_PLANS = ["free", "starter", "professional", "enterprise"]

TICKET_SUBJECTS = {
    "billing": [
        "Unexpected charge on my account",
        "Need to update payment method",
        "Invoice discrepancy for last month",
        "Requesting refund for duplicate charge",
        "Subscription downgrade not reflected in billing",
        "Cannot find my latest invoice",
        "Tax exemption not applied",
        "Trial period ended but I wasn't notified",
    ],
    "technical": [
        "API returning 500 errors intermittently",
        "Login fails with SSO enabled",
        "Data export not completing",
        "Dashboard loading extremely slowly",
        "Webhook delivery failures",
        "Cannot connect to database integration",
        "File upload failing for large files",
        "Search function returning wrong results",
    ],
    "account": [
        "Need to change my account email",
        "Two-factor authentication locked me out",
        "Request to delete my account",
        "Cannot access team workspace",
        "Invited team member didn't receive email",
        "Permission issue on shared resources",
        "Account merge request",
        "Need to transfer account ownership",
    ],
    "general": [
        "How do I get started with the API?",
        "What's the difference between plans?",
        "Is there a mobile app available?",
        "Do you support SAML SSO?",
        "What are your SLA guarantees?",
        "Can I get a demo of enterprise features?",
        "How do I set up webhooks?",
        "What data centers do you use?",
    ],
    "feature_request": [
        "Can you add Slack integration?",
        "Need dark mode in the dashboard",
        "Request: bulk import from CSV",
        "Would love Jira integration",
        "Can we get audit logs for all actions?",
        "Need role-based access control",
        "Request: scheduled reports",
        "Would like API rate limit dashboard",
    ],
}

KB_ARTICLES = {
    "getting_started": [
        {"id": "KB-001", "title": "Quick Start Guide", "summary": "Get up and running in 5 minutes with our step-by-step guide."},
        {"id": "KB-002", "title": "Account Setup", "summary": "How to configure your account settings, team, and preferences."},
        {"id": "KB-003", "title": "API Authentication", "summary": "Learn how to authenticate API requests using API keys or OAuth."},
    ],
    "troubleshooting": [
        {"id": "KB-010", "title": "Common Login Issues", "summary": "Solutions for SSO, password reset, and 2FA problems."},
        {"id": "KB-011", "title": "API Error Codes", "summary": "Complete reference of error codes and their solutions."},
        {"id": "KB-012", "title": "Performance Troubleshooting", "summary": "Steps to diagnose and fix slow performance."},
        {"id": "KB-013", "title": "Data Export Issues", "summary": "How to resolve failed or incomplete data exports."},
    ],
    "billing": [
        {"id": "KB-020", "title": "Understanding Your Invoice", "summary": "Breakdown of invoice line items and billing cycles."},
        {"id": "KB-021", "title": "Changing Your Plan", "summary": "How to upgrade, downgrade, or cancel your subscription."},
        {"id": "KB-022", "title": "Payment Methods", "summary": "Supported payment methods and how to update billing info."},
    ],
    "api": [
        {"id": "KB-030", "title": "API Rate Limits", "summary": "Understanding rate limits and how to handle 429 responses."},
        {"id": "KB-031", "title": "Webhook Configuration", "summary": "Setting up and testing webhook endpoints."},
        {"id": "KB-032", "title": "Pagination Guide", "summary": "How to use cursor-based pagination in API responses."},
    ],
    "integrations": [
        {"id": "KB-040", "title": "Slack Integration", "summary": "Connect your workspace with Slack for real-time notifications."},
        {"id": "KB-041", "title": "Jira Integration", "summary": "Sync tickets between your account and Jira projects."},
        {"id": "KB-042", "title": "Database Connectors", "summary": "Connect to PostgreSQL, MySQL, MongoDB, and more."},
    ],
}

TICKET_STATUSES = ["open", "in_progress", "pending", "resolved", "closed"]
TICKET_PRIORITIES = ["low", "medium", "high", "critical"]
TEAMS = ["senior_support", "engineering", "billing", "security"]

ESCALATION_REASONS = [
    "Customer is a high-value enterprise account and issue is impacting production",
    "Issue persists after standard troubleshooting steps",
    "Potential security vulnerability reported",
    "Data loss or corruption suspected",
    "SLA breach risk — ticket has been open for over 24 hours",
    "Customer explicitly requested manager escalation",
    "Issue requires engineering investigation — not resolvable at L1",
    "Billing discrepancy exceeds $500 and needs finance review",
]

# Out-of-scope requests
OUT_OF_SCOPE_REQUESTS = [
    "Can you build a custom feature for me right now?",
    "Write me a Python script to use your API",
    "What's the best CRM software?",
    "Can you access my competitor's account?",
    "Help me hack into another account",
    "Can you give me a free enterprise license?",
    "What are your employees' salaries?",
    "Can you recommend a good restaurant nearby?",
    "Write my resume for me",
    "What's the weather like today?",
]


def get_customer_id(rng) -> str:
    return f"CUST-{rng.randint(10000, 99999)}"


def get_ticket_id(rng) -> str:
    return f"TKT-{rng.randint(10000, 99999)}"


def get_customer_name(rng) -> str:
    return f"{rng.choice(CUSTOMER_FIRST_NAMES)} {rng.choice(CUSTOMER_LAST_NAMES)}"


def get_customer_data(rng) -> dict:
    name = get_customer_name(rng)
    cid = get_customer_id(rng)
    return {
        "customer_id": cid,
        "name": name,
        "email": fake_email(rng, name),
        "plan": rng.choice(SUBSCRIPTION_PLANS),
        "product": rng.choice(PRODUCT_NAMES),
        "created": fake_date(rng, year=rng.randint(2022, 2025)),
        "tickets_count": rng.randint(0, 15),
        "mrr": round(rng.uniform(0, 500), 2),
    }


def get_ticket_data(rng, customer_id: str | None = None, category: str | None = None) -> dict:
    if category is None:
        category = rng.choice(list(TICKET_SUBJECTS.keys()))
    subject = rng.choice(TICKET_SUBJECTS[category])
    return {
        "ticket_id": get_ticket_id(rng),
        "subject": subject,
        "category": category,
        "status": rng.choice(TICKET_STATUSES),
        "priority": rng.choice(TICKET_PRIORITIES),
        "customer_id": customer_id or get_customer_id(rng),
        "created": fake_date(rng, year=2025),
        "assignee": rng.choice(["Agent Smith", "Agent Lee", "Agent Patel", "Unassigned"]),
    }
