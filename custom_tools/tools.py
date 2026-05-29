from src.masai.Tools.Tool import tool
from typing import List, Dict, Optional

@tool("analyze_stock_portfolio")
def analyze_stock_portfolio(portfolio: Dict[str, float], risk_tolerance: str, investment_horizon_years: int) -> str:
    """
    Analyzes a given stock portfolio and provides a comprehensive risk assessment and projection.
    
    This tool performs advanced quantitative analysis on a user's stock holdings. It takes the current 
    portfolio allocation, evaluates it against historical market data, and projects future performance 
    based on the specified investment horizon and risk tolerance. It also suggests rebalancing strategies 
    to optimize the Sharpe ratio.
    
    Args:
        portfolio: A dictionary mapping stock ticker symbols (strings) to the percentage allocation (float). Must sum to 100.
        risk_tolerance: The user's risk tolerance level. Must be one of: 'low', 'medium', 'high'.
        investment_horizon_years: The number of years the user plans to hold the investments.
    """
    return f"Portfolio analyzed. With a {risk_tolerance} risk tolerance over {investment_horizon_years} years, the expected annualized return is 8.5%."

@tool("schedule_complex_meeting")
def schedule_complex_meeting(participants: List[str], duration_minutes: int, timezone: str, optional_attendees: Optional[List[str]] = None) -> str:
    """
    Finds optimal meeting times across multiple timezones for a group of participants.
    
    This scheduling tool cross-references the calendars of all mandatory and optional participants 
    to find a time slot that satisfies the duration requirement within standard working hours 
    for the specified timezone. It automatically handles daylight saving time adjustments and 
    prioritizes slots where optional attendees are also free.
    
    Args:
        participants: A list of email addresses of mandatory participants.
        duration_minutes: The length of the meeting in minutes.
        timezone: The primary timezone for the meeting organizer (e.g., 'America/New_York').
        optional_attendees: A list of email addresses of optional participants.
    """
    return f"Meeting scheduled for {duration_minutes} minutes in {timezone} with {len(participants)} mandatory participants."

@tool("generate_marketing_copy")
def generate_marketing_copy(product_name: str, target_audience: str, key_benefits: List[str], tone: str) -> str:
    """
    Generates high-converting marketing copy for a product tailored to a specific audience.
    
    This tool utilizes advanced natural language generation to craft compelling ad copy, 
    email campaigns, and social media posts. It incorporates psychological triggers and 
    persuasive writing techniques matched to the specified tone to maximize engagement 
    and conversion rates for the target demographic.
    
    Args:
        product_name: The name of the product or service being marketed.
        target_audience: A description of the ideal customer profile.
        key_benefits: A list of the primary value propositions of the product.
        tone: The desired tone of the copy (e.g., 'professional', 'humorous', 'urgent').
    """
    benefits_str = ", ".join(key_benefits)
    return f"Generated {tone} marketing copy for {product_name} targeting {target_audience}. Highlighting: {benefits_str}."

@tool("calculate_rocket_trajectory")
def calculate_rocket_trajectory(payload_mass_kg: float, target_orbit_altitude_km: float, launch_site_latitude: float) -> str:
    """
    Calculates the required orbital insertion parameters for a rocket launch.
    
    This tool performs orbital mechanics simulations to determine the delta-v, required fuel mass, 
    and optimal launch window for a specific payload aiming for a target circular orbit. It accounts 
    for the Earth's rotation based on the launch site latitude and models atmospheric drag during ascent.
    
    Args:
        payload_mass_kg: The mass of the payload in kilograms.
        target_orbit_altitude_km: The desired altitude of the circular orbit in kilometers.
        launch_site_latitude: The latitude of the launch facility in degrees.
    """
    return f"Trajectory calculated: Required Delta-V is 9400 m/s for a {payload_mass_kg}kg payload to reach {target_orbit_altitude_km}km orbit."
