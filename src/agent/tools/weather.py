# Weather API tool
from __future__ import annotations

import os

import httpx
from httpx import HTTPStatusError, RequestError
from langchain.tools import tool

from agent.common.logging_config import get_logger

logger = get_logger(__name__)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

WeatherToolReturn = tuple[str, str, str, str]

@tool
def get_weather(city: str, units: str) -> WeatherToolReturn:
    """
    This ia a weather tool to get weather data from OpenWeather
    based on the user's location (city name).
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": units
    }
    try:
        response = httpx.get(url, params=params)
        response.raise_for_status()

        # Break down response into strings that are useful
        data = response.json()
        description = data["weather"][0]["description"]
        temp_real = data["main"]["temp"]
        temp_max = data["main"]["temp_max"]
        temp_min = data["main"]["temp_min"]
        humidity = data["main"]["humidity"]
        cloudines = data["clouds"]["all"]

        unit_symbol = "°C" if units == "metric" else "°F"

        return (
            f"Weather in {city}: {description} ",
            f"Temperature: {temp_real} {unit_symbol} with maximum and minimum of {temp_max} {unit_symbol} and {temp_min} {unit_symbol} ",
            f"Humidity: {humidity} % ",
            f"Cloudines: {cloudines} "
        )
    except HTTPStatusError as e:
        logger.error(f"Error fetching data: {e}")
        raise e
    except RequestError as e:
        logger.error(f"Error fetching data: {e}")
        raise e

