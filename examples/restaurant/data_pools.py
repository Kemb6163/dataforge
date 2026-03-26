"""Restaurant-specific data pools for generating realistic examples."""

from __future__ import annotations

from dataforge.core.rng import make_rng
from dataforge.generation.pools import fake_name, fake_id, fake_date, fake_time, fake_price

# Menu items by category
MENU_ITEMS = {
    "appetizers": [
        {"id": "app-001", "name": "Bruschetta", "price": 9.99, "dietary": ["vegetarian"], "calories": 280},
        {"id": "app-002", "name": "Calamari Fritti", "price": 12.99, "dietary": [], "calories": 420},
        {"id": "app-003", "name": "Caprese Salad", "price": 10.99, "dietary": ["vegetarian", "gluten-free"], "calories": 220},
        {"id": "app-004", "name": "Stuffed Mushrooms", "price": 11.99, "dietary": ["vegetarian"], "calories": 310},
        {"id": "app-005", "name": "Edamame", "price": 7.99, "dietary": ["vegan", "gluten-free"], "calories": 180},
        {"id": "app-006", "name": "Spring Rolls", "price": 8.99, "dietary": ["vegan"], "calories": 260},
        {"id": "app-007", "name": "Shrimp Cocktail", "price": 14.99, "dietary": ["gluten-free"], "calories": 200},
        {"id": "app-008", "name": "Hummus Platter", "price": 9.49, "dietary": ["vegan", "gluten-free"], "calories": 340},
    ],
    "mains": [
        {"id": "main-001", "name": "Grilled Salmon", "price": 24.99, "dietary": ["gluten-free"], "calories": 520},
        {"id": "main-002", "name": "Margherita Pizza", "price": 16.99, "dietary": ["vegetarian"], "calories": 680},
        {"id": "main-003", "name": "Chicken Parmesan", "price": 19.99, "dietary": [], "calories": 750},
        {"id": "main-004", "name": "Vegetable Stir-Fry", "price": 15.99, "dietary": ["vegan", "gluten-free"], "calories": 380},
        {"id": "main-005", "name": "Ribeye Steak", "price": 32.99, "dietary": ["gluten-free"], "calories": 820},
        {"id": "main-006", "name": "Mushroom Risotto", "price": 18.99, "dietary": ["vegetarian", "gluten-free"], "calories": 550},
        {"id": "main-007", "name": "Fish and Chips", "price": 17.99, "dietary": [], "calories": 780},
        {"id": "main-008", "name": "Pasta Primavera", "price": 16.49, "dietary": ["vegetarian"], "calories": 490},
        {"id": "main-009", "name": "Lamb Chops", "price": 29.99, "dietary": ["gluten-free"], "calories": 640},
        {"id": "main-010", "name": "Thai Green Curry", "price": 18.49, "dietary": ["gluten-free"], "calories": 560},
    ],
    "desserts": [
        {"id": "des-001", "name": "Tiramisu", "price": 9.99, "dietary": ["vegetarian"], "calories": 420},
        {"id": "des-002", "name": "Chocolate Lava Cake", "price": 11.99, "dietary": ["vegetarian"], "calories": 560},
        {"id": "des-003", "name": "Crème Brûlée", "price": 10.99, "dietary": ["vegetarian", "gluten-free"], "calories": 380},
        {"id": "des-004", "name": "Fruit Sorbet", "price": 7.99, "dietary": ["vegan", "gluten-free"], "calories": 150},
        {"id": "des-005", "name": "Cheesecake", "price": 10.49, "dietary": ["vegetarian"], "calories": 480},
        {"id": "des-006", "name": "Panna Cotta", "price": 9.49, "dietary": ["vegetarian", "gluten-free"], "calories": 320},
    ],
    "drinks": [
        {"id": "drk-001", "name": "Espresso", "price": 3.99, "dietary": ["vegan", "gluten-free"], "calories": 5},
        {"id": "drk-002", "name": "Fresh Lemonade", "price": 4.99, "dietary": ["vegan", "gluten-free"], "calories": 120},
        {"id": "drk-003", "name": "House Red Wine", "price": 9.99, "dietary": ["vegan", "gluten-free"], "calories": 125},
        {"id": "drk-004", "name": "Craft Beer", "price": 7.99, "dietary": ["vegan"], "calories": 180},
        {"id": "drk-005", "name": "Sparkling Water", "price": 2.99, "dietary": ["vegan", "gluten-free"], "calories": 0},
        {"id": "drk-006", "name": "Iced Tea", "price": 3.49, "dietary": ["vegan", "gluten-free"], "calories": 80},
    ],
    "sides": [
        {"id": "side-001", "name": "Garlic Bread", "price": 5.99, "dietary": ["vegetarian"], "calories": 280},
        {"id": "side-002", "name": "Caesar Salad", "price": 8.99, "dietary": [], "calories": 320},
        {"id": "side-003", "name": "Sweet Potato Fries", "price": 6.99, "dietary": ["vegan", "gluten-free"], "calories": 350},
        {"id": "side-004", "name": "Steamed Vegetables", "price": 5.49, "dietary": ["vegan", "gluten-free"], "calories": 120},
        {"id": "side-005", "name": "Truffle Mac & Cheese", "price": 9.99, "dietary": ["vegetarian"], "calories": 520},
    ],
}

ALL_DISHES = []
for category, items in MENU_ITEMS.items():
    for item in items:
        ALL_DISHES.append({**item, "category": category})

DISH_BY_ID = {d["id"]: d for d in ALL_DISHES}

# Query templates for menu search
MENU_QUERIES = [
    "What vegetarian options do you have?",
    "Show me your dessert menu",
    "What's gluten-free?",
    "I'm looking for something under ${max_price}",
    "Do you have any vegan dishes?",
    "What appetizers do you recommend?",
    "Can I see your pasta options?",
    "What are your most popular mains?",
    "Any dairy-free desserts?",
    "What soups do you have today?",
    "I want something light, what do you suggest?",
    "What's your spiciest dish?",
    "Show me fish and seafood options",
    "What sides go well with steak?",
    "Do you have any specials today?",
    "What drinks do you have?",
    "I'm looking for a kids meal",
    "What can I get for under $20?",
    "Any nut-free options?",
    "What's your chef's recommendation?",
]

# Customer name templates
CUSTOMER_FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Linda",
    "William", "Elizabeth", "Richard", "Susan", "Thomas", "Jessica", "Daniel", "Sarah",
    "Christopher", "Karen", "Matthew", "Nancy", "Anthony", "Lisa", "Mark", "Betty",
    "Charles", "Dorothy", "Steven", "Margaret", "Paul", "Sandra",
]

CUSTOMER_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]

SPECIAL_REQUESTS = [
    "We'll need a high chair for a toddler",
    "Wheelchair accessible seating please",
    "Window seat if possible",
    "Can we have a booth?",
    "Birthday celebration — can you arrange a cake?",
    "Business dinner, quiet area preferred",
    "We'd like to sit on the patio",
    "Anniversary dinner, something romantic please",
    "",
    "",
    "",
    "",  # empty = no special request (weighted towards no request)
]

ORDER_STATUSES = [
    {"status": "received", "eta_minutes": 25, "message": "Your order has been received and is being prepared."},
    {"status": "preparing", "eta_minutes": 18, "message": "Your food is currently being prepared by the chef."},
    {"status": "cooking", "eta_minutes": 12, "message": "Your dishes are being cooked now."},
    {"status": "plating", "eta_minutes": 5, "message": "Your order is being plated and will be served shortly."},
    {"status": "served", "eta_minutes": 0, "message": "Your order has been served. Enjoy your meal!"},
    {"status": "cancelled", "eta_minutes": 0, "message": "This order was cancelled."},
]

REVIEW_PROMPTS = [
    "The {dish} was absolutely {adj}. {detail}",
    "We had a {experience} evening. {detail}",
    "I'd give the {dish} a {rating}/5. {detail}",
    "The service was {service_quality}. {detail}",
    "First time here and {impression}. {detail}",
]

REVIEW_ADJECTIVES = ["delicious", "outstanding", "amazing", "superb", "mediocre", "disappointing", "okay", "excellent"]
EXPERIENCE_ADJECTIVES = ["wonderful", "lovely", "pleasant", "fantastic", "decent", "mixed", "memorable"]
SERVICE_QUALITIES = ["excellent", "attentive", "prompt", "friendly", "a bit slow", "outstanding", "adequate"]

# Out-of-scope requests (for no-tool restraint training)
OUT_OF_SCOPE_REQUESTS = [
    "Can you call me a taxi?",
    "What's the weather like today?",
    "Can you recommend a movie?",
    "What time does the mall close?",
    "Can you help me with my math homework?",
    "Who won the game last night?",
    "Can you translate this to French?",
    "What's the stock price of Apple?",
    "Can you set a reminder for me?",
    "What are the best hotels nearby?",
    "Can you play some music?",
    "How do I get to the airport?",
    "Can you book me a flight?",
    "What's the population of Tokyo?",
    "Can you help me write a poem?",
    "What are the news headlines today?",
    "Can you recommend a doctor?",
    "How do I fix my computer?",
    "Can you order me an Uber?",
    "What's the exchange rate for euros?",
]

# Polite refusal templates for out-of-scope
REFUSAL_TEMPLATES = [
    "I appreciate your question, but I'm specifically designed to help with restaurant-related tasks like searching our menu, making reservations, checking order status, and handling reviews. For {topic}, you might want to try {suggestion}.",
    "That's a great question, but it's outside my area of expertise. I'm here to help with anything related to our restaurant — menu items, reservations, orders, and reviews. Is there anything restaurant-related I can assist you with?",
    "I'm afraid I can't help with that since I'm a restaurant assistant. I can help you find dishes on our menu, make a reservation, check your order, or submit a review. Would you like to do any of those?",
    "Sorry, that's not something I can assist with. My capabilities are focused on our restaurant services. I'd be happy to help you browse the menu, book a table, or check on an order though!",
]


def get_random_dish(rng, category: str | None = None) -> dict:
    """Get a random dish, optionally from a specific category."""
    if category and category in MENU_ITEMS:
        return rng.choice(MENU_ITEMS[category])
    return rng.choice(ALL_DISHES)


def get_random_dishes(rng, n: int = 3, category: str | None = None) -> list[dict]:
    """Get n random dishes."""
    pool = MENU_ITEMS.get(category, ALL_DISHES) if category else ALL_DISHES
    return rng.sample(pool, min(n, len(pool)))


def get_customer_name(rng) -> str:
    """Generate a random customer name."""
    return f"{rng.choice(CUSTOMER_FIRST_NAMES)} {rng.choice(CUSTOMER_LAST_NAMES)}"


def get_order_id(rng) -> str:
    """Generate a random order ID."""
    return f"ORD-{rng.randint(10000, 99999)}"


def get_reservation_date(rng) -> str:
    """Generate a random future date."""
    return fake_date(rng, year=2025, month_range=(3, 12))


def get_reservation_time(rng) -> str:
    """Generate a random dinner time."""
    return fake_time(rng, hour_range=(11, 21))


def get_party_size(rng) -> int:
    """Generate a random party size (weighted towards 2-4)."""
    weights = [5, 25, 30, 20, 10, 5, 3, 2]  # for sizes 1-8
    sizes = list(range(1, 9))
    return rng.choices(sizes, weights=weights, k=1)[0]
