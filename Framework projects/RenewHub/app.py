from flask import Flask, render_template, request, jsonify
from dataclasses import dataclass
from typing import List, Optional
import requests
import logging
import os
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    API_KEY = os.getenv("WEATHER_API_KEY")
    WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

@dataclass
class CostBenefitAnalysis:
    installation_cost: float
    annual_maintenance: float
    annual_savings: float
    payback_period: float
    carbon_reduction: float
    government_incentives: float
    lifetime: int
    roi: float

@dataclass
class EnergySource:
    name: str
    efficiency: float
    description: str
    cost_benefit: Optional[CostBenefitAnalysis] = None

@dataclass
class WeatherData:
    temperature: float
    wind_speed: float
    rainfall: float  # in mm
    cloud_cover: int  # in percentage
    condition: str
    humidity: int


class WeatherAPIError(Exception):
    # Custom exception for weather API related errors
    pass


def get_weather_data(city: str) -> WeatherData:

    # Fetch weather data from WeatherAPI.com with error handling

    try:
        params = {"key": Config.API_KEY, "q": city, "aqi": "no"}
        response = requests.get(Config.WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        return WeatherData(
            temperature=data["current"]["temp_c"],
            wind_speed=data["current"]["wind_kph"] * 0.27778,  # Convert kph to m/s
            rainfall=data["current"].get("precip_mm", 0),
            cloud_cover=data["current"]["cloud"],
            condition=data["current"]["condition"]["text"].lower(),
            humidity=data["current"]["humidity"],
        )
    except requests.RequestException as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        raise WeatherAPIError(f"Failed to fetch weather data: {str(e)}")


def calculate_cost_benefit(
    energy_source: str,
    efficiency: float,
    power_requirement: float,  # in kW
    electricity_rate: float,  # INR per kWh
    location: str,
) -> CostBenefitAnalysis:
    """Calculate cost-benefit analysis for a given energy source."""

    # Base calculations adjusted by efficiency
    effective_power = power_requirement * (efficiency / 100)

    # Cost parameters based on energy source type (2024 estimates)
    costs = {
        "Solar Energy": {
            "cost_per_kw": 45000,  # INR per kW installed
            "maintenance_factor": 0.01,  # 1% of installation cost
            "lifetime": 25,
            "carbon_factor": 0.85,  # CO2 reduction factor
        },
        "Wind Energy": {
            "cost_per_kw": 60000,
            "maintenance_factor": 0.02,
            "lifetime": 20,
            "carbon_factor": 0.90,
        },
        "Hydropower": {
            "cost_per_kw": 70000,
            "maintenance_factor": 0.025,
            "lifetime": 30,
            "carbon_factor": 0.95,
        },
        "Geothermal Energy": {
            "cost_per_kw": 80000,
            "maintenance_factor": 0.015,
            "lifetime": 25,
            "carbon_factor": 0.92,
        },
    }

    source_params = costs.get(energy_source)
    if not source_params:
        raise ValueError(f"Unknown energy source: {energy_source}")

    # Calculate installation cost
    installation_cost = effective_power * source_params["cost_per_kw"]

    # Calculate annual maintenance
    annual_maintenance = installation_cost * source_params["maintenance_factor"]

    # Calculate annual energy production (kWh)
    capacity_factor = efficiency / 100
    annual_production = effective_power * 8760 * capacity_factor  # 8760 hours in a year

    # Calculate annual savings
    annual_savings = annual_production * electricity_rate

    # Calculate government incentives (based on location and type)
    government_incentives = calculate_incentives(
        energy_source, installation_cost, location
    )

    # Calculate payback period
    net_installation_cost = installation_cost - government_incentives
    annual_net_savings = annual_savings - annual_maintenance
    payback_period = (
        net_installation_cost / annual_net_savings
        if annual_net_savings > 0
        else float("inf")
    )

    # Calculate ROI
    lifetime_savings = annual_net_savings * source_params["lifetime"]
    roi = ((lifetime_savings - net_installation_cost) / net_installation_cost) * 100

    # Calculate carbon reduction
    carbon_reduction = (
        annual_production * 0.4 * source_params["carbon_factor"]
    )  # 0.4 metric tons CO2 per MWh

    return CostBenefitAnalysis(
        installation_cost=round(installation_cost, 2),
        annual_maintenance=round(annual_maintenance, 2),
        annual_savings=round(annual_savings, 2),
        payback_period=round(payback_period, 1),
        carbon_reduction=round(carbon_reduction, 1),
        government_incentives=round(government_incentives, 2),
        lifetime=source_params["lifetime"],
        roi=round(roi, 1),
    )


def calculate_incentives(
    energy_source: str, installation_cost: float, location: str
) -> float:
    """Calculate available government incentives based on location and energy type."""
    # Basic incentive calculations (can be expanded with more detailed location-specific data)
    base_incentive_rate = {
        "Solar Energy": 0.30,  # 30% federal tax credit
        "Wind Energy": 0.30,
        "Hydropower": 0.25,
        "Geothermal Energy": 0.25,
    }

    # Additional state-specific incentives could be added here
    state_incentives = {
        "Gujarat": {
            "Solar Energy": 0.10,
            "Wind Energy": 0.12,
        },
        "Tamil Nadu": {
            "Solar Energy": 0.08,
            "Wind Energy": 0.15,
        },
        "Maharashtra": {
            "Solar Energy": 0.05,
            "Wind Energy": 0.08,
        },
        "Karnataka": {
            "Solar Energy": 0.07,
            "Wind Energy": 0.10,
        },
        "Rajasthan": {
            "Solar Energy": 0.12,
            "Wind Energy": 0.09,
        },
        "Andhra Pradesh": {
            "Solar Energy": 0.08,
            "Wind Energy": 0.11,
        },
        "Madhya Pradesh": {
            "Solar Energy": 0.09,
            "Wind Energy": 0.07,
        },
        "Telangana": {
            "Solar Energy": 0.08,
            "Wind Energy": 0.06,
        },
        "Kerala": {
            "Solar Energy": 0.04,
            "Wind Energy": 0.05,
        },
        "Uttar Pradesh": {
            "Solar Energy": 0.06,
            "Wind Energy": 0.04,
        },
        "Bihar": {
            "Solar Energy": 0.05,
            "Wind Energy": 0.03,
        },
        "West Bengal": {
            "Solar Energy": 0.04,
            "Wind Energy": 0.05,
        },
        "Odisha": {
            "Solar Energy": 0.05,
            "Wind Energy": 0.06,
        },
        "Punjab": {
            "Solar Energy": 0.06,
            "Wind Energy": 0.03,
        },
        "Haryana": {
            "Solar Energy": 0.05,
            "Wind Energy": 0.04,
        },
        "Chhattisgarh": {
            "Solar Energy": 0.06,
            "Wind Energy": 0.04,
        },
        "Jharkhand": {
            "Solar Energy": 0.04,
            "Wind Energy": 0.03,
        },
        "Uttarakhand": {
            "Solar Energy": 0.03,
            "Wind Energy": 0.04,
        },
        "Himachal Pradesh": {
            "Solar Energy": 0.02,
            "Wind Energy": 0.05,
        },
        "Assam": {
            "Solar Energy": 0.03,
            "Wind Energy": 0.02,
        },
        "Goa": {
            "Solar Energy": 0.02,
            "Wind Energy": 0.03,
        },
    }

    base_rate = base_incentive_rate.get(energy_source, 0)
    state_rate = state_incentives.get(location[:2].upper(), 0)

    return installation_cost * (base_rate + state_rate)


def predict_energy_source(
    weather: WeatherData,
    sunshine_hours: int,
    is_near_water: bool = False,
    is_geothermal_region: bool = False,
    power_requirement: float = 10.0,  # Default 10kW system
    electricity_rate: float = 0.12,  # Default $0.12 per kWh
    state: str = "",
) -> List[EnergySource]:
    # Predict suitable energy sources with detailed efficiency calculations
    energy_sources = []

    # Solar Energy Assessment
    solar_efficiency = calculate_solar_efficiency(weather, sunshine_hours)
    if solar_efficiency > 30:
        cost_benefit = calculate_cost_benefit(
            "Solar Energy", solar_efficiency, power_requirement, electricity_rate, state
        )
        energy_sources.append(
            EnergySource(
                name="Solar Energy",
                efficiency=solar_efficiency,
                description=f"Suitable with {solar_efficiency:.1f}% efficiency based on {sunshine_hours}h sunshine and {weather.cloud_cover}% cloud cover",
                cost_benefit=cost_benefit,
            )
        )

    # Wind Energy Assessment
    wind_efficiency = calculate_wind_efficiency(weather.wind_speed)
    if wind_efficiency > 20:
        cost_benefit = calculate_cost_benefit(
            "Wind Energy", wind_efficiency, power_requirement, electricity_rate, state
        )
        energy_sources.append(
            EnergySource(
                name="Wind Energy",
                efficiency=wind_efficiency,
                description=f"Viable with {wind_efficiency:.1f}% efficiency at {weather.wind_speed:.1f} m/s wind speed",
                cost_benefit=cost_benefit,
            )
        )

    # Hydropower Assessment
    if is_near_water:
        hydro_efficiency = calculate_hydro_efficiency(weather.rainfall)
        cost_benefit = calculate_cost_benefit(
            "Hydropower", hydro_efficiency, power_requirement, electricity_rate, state
        )
        energy_sources.append(
            EnergySource(
                name="Hydropower",
                efficiency=hydro_efficiency,
                description=f"Feasible with {hydro_efficiency:.1f}% efficiency given water proximity and {weather.rainfall}mm rainfall",
                cost_benefit=cost_benefit,
            )
        )

    # Geothermal Assessment
    if is_geothermal_region:
        geothermal_efficiency = 85.0
        cost_benefit = calculate_cost_benefit(
            "Geothermal Energy",
            geothermal_efficiency,
            power_requirement,
            electricity_rate,
            state,
        )
        energy_sources.append(
            EnergySource(
                name="Geothermal Energy",
                efficiency=geothermal_efficiency,
                description="Highly efficient option in geothermal region",
                cost_benefit=cost_benefit,
            )
        )

    return energy_sources


def calculate_solar_efficiency(weather: WeatherData, sunshine_hours: int) -> float:
    # Calculate solar energy efficiency based on weather conditions
    base_efficiency = 40  # Base efficiency for modern solar panels

    # Reduce efficiency based on cloud cover
    cloud_factor = 1 - (weather.cloud_cover / 100) * 0.7
    # Adjust for sunshine hours
    sunshine_factor = min(sunshine_hours / 12, 1)  # Normalize to 12 hours max
    # Temperature adjustment (efficiency drops above 25°C)
    temp_factor = 1 - max(0, (weather.temperature - 25) * 0.004)
    return base_efficiency * cloud_factor * sunshine_factor * temp_factor


def calculate_wind_efficiency(wind_speed: float) -> float:
    # Calculate wind energy efficiency based on wind speed
    if wind_speed < 3:  # Cut-in speed
        return 0
    elif wind_speed > 25:  # Cut-out speed
        return 0
    else:
        # Simplified efficiency curve
        return min(90, wind_speed * 5)


def calculate_hydro_efficiency(rainfall: float) -> float:
    # Calculate hydropower efficiency based on rainfall
    base_efficiency = 70  # Base efficiency for hydropower
    rainfall_factor = min(1 + (rainfall / 100), 1.3)  # Max 30% boost from rainfall
    return base_efficiency * rainfall_factor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Input validation
        if not request.form.get("city"):
            return jsonify({"error": "City is required"}), 400

        city = request.form["city"]

        if not request.form.get("state"):
            return jsonify({"error": "State is required"}), 400
        state = request.form["state"]
        try:
            sunshine_hours = int(request.form["sunshine_hours"])
            if not (0 <= sunshine_hours <= 24):
                return (
                    jsonify({"error": "Sunshine hours must be between 0 and 24"}),
                    400,
                )
        except ValueError:
            return jsonify({"error": "Invalid sunshine hours value"}), 400

        # New parameters for cost-benefit analysis
        power_requirement = float(request.form.get("power_requirement", 10.0))
        electricity_rate = float(request.form.get("electricity_rate", 0.12))
        is_near_water = request.form.get("is_near_water") == "on"
        is_geothermal_region = request.form.get("is_geothermal_region") == "on"
        weather_data = get_weather_data(city)

        energy_sources = predict_energy_source(
            weather_data,
            sunshine_hours,
            is_near_water,
            is_geothermal_region,
            power_requirement,
            electricity_rate,
            state,
        )
        # Formating  response
        response = {
            "timestamp": datetime.now().isoformat(),
            "city": city,
            "state": state,
            "weather": {
                "temperature": weather_data.temperature,
                "wind_speed": weather_data.wind_speed,
                "condition": weather_data.condition,
                "cloud_cover": weather_data.cloud_cover,
            },
            "energy_sources": [
                {
                    "name": source.name,
                    "efficiency": source.efficiency,
                    "description": source.description,
                    "cost_benefit": (
                        {
                            "installation_cost": source.cost_benefit.installation_cost,
                            "annual_maintenance": source.cost_benefit.annual_maintenance,
                            "annual_savings": source.cost_benefit.annual_savings,
                            "payback_period": source.cost_benefit.payback_period,
                            "carbon_reduction": source.cost_benefit.carbon_reduction,
                            "government_incentives": source.cost_benefit.government_incentives,
                            "lifetime": source.cost_benefit.lifetime,
                            "roi": source.cost_benefit.roi,
                        }
                        if source.cost_benefit
                        else None
                    ),
                }
                for source in energy_sources
            ],
        }
        if not energy_sources:
            response["message"] = (
                "No optimal energy sources found for the given conditions"
            )
        return jsonify(response)

    except WeatherAPIError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


if __name__ == "__main__":
    app.run(debug=True)
