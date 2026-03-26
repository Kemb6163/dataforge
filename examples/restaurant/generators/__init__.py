"""Restaurant example generators.

Each generator produces training examples for a specific conversation pattern:
- menu_search: Single tool calls (search_menu, get_dish_details)
- reservations: Multi-turn conversations (check_availability + make_reservation)
- order_management: Parallel tool calls (multiple get_order_status)
- reviews: No-tool restraint (out-of-scope handling + review submission)
- complex_scenarios: Multi-tool chains (search → detail → reserve → confirm)
- dpo_pairs: DPO preference pairs
"""

from examples.restaurant.generators.menu_search import MenuSearchGenerator
from examples.restaurant.generators.reservations import ReservationGenerator
from examples.restaurant.generators.order_management import OrderManagementGenerator
from examples.restaurant.generators.reviews import ReviewGenerator
from examples.restaurant.generators.complex_scenarios import ComplexScenarioGenerator
from examples.restaurant.generators.dpo_pairs import RestaurantDPOGenerator

__all__ = [
    "MenuSearchGenerator",
    "ReservationGenerator",
    "OrderManagementGenerator",
    "ReviewGenerator",
    "ComplexScenarioGenerator",
    "RestaurantDPOGenerator",
]
