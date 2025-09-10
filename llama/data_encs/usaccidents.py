import pdb
from datetime import datetime
from enum import Enum

import pandas as pd
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


OUTCOMES = ["Severity"]

"""COLUMNS"""
# ------------------------------------------------------------------
#  Datetime columns
# ------------------------------------------------------------------


class ColumnsEncoding(Enum):
    start_time_col = ColumnToText(
        "Start_Time",
        short_description="start timestamp of the accident",
        value_map=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
        if pd.notna(x)
        else "missing",
    )

    end_time_col = ColumnToText(
        "End_Time",
        short_description="end timestamp of the accident",
        value_map=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
        if pd.notna(x)
        else "missing",
    )

    # ------------------------------------------------------------------
    #  Geo‑location & distance
    # ------------------------------------------------------------------
    start_lat_col = ColumnToText(
        "Start_Lat",
        short_description="starting latitude",
        value_map=lambda x: f"{x:.5f}" if pd.notna(x) else "missing",
    )

    start_lng_col = ColumnToText(
        "Start_Lng",
        short_description="starting longitude",
        value_map=lambda x: f"{x:.5f}" if pd.notna(x) else "missing",
    )

    distance_col = ColumnToText(
        "Distance(mi)",
        short_description="road distance affected (miles)",
        value_map=lambda x: f"{x} mi" if pd.notna(x) else "missing",
    )

    # ------------------------------------------------------------------
    #  Free‑text location / address fields
    # ------------------------------------------------------------------
    description_col = ColumnToText(
        "Description",
        short_description="narrative description of the accident",
        value_map=lambda x: x if isinstance(x, str) else "no description",
    )

    street_col = ColumnToText(
        "Street",
        short_description="street name",
        value_map=lambda x: f"on {x}" if isinstance(x, str) else "unknown street",
    )

    city_col = ColumnToText(
        "City", short_description="city", value_map=lambda x: x if isinstance(x, str) else "unknown city"
    )

    county_col = ColumnToText(
        "County",
        short_description="county",
        value_map=lambda x: f"{x} County" if isinstance(x, str) else "unknown county",
    )

    state_col = ColumnToText(
        "State",
        short_description="state abbreviation",
        value_map=lambda x: x if isinstance(x, str) else "unknown state",
    )

    zipcode_col = ColumnToText(
        "Zipcode", short_description="ZIP code", value_map=lambda x: str(x) if pd.notna(x) else "unknown ZIP"
    )

    country_col = ColumnToText(
        "Country", short_description="country code", value_map=lambda x: x if isinstance(x, str) else "unknown country"
    )

    # ------------------------------------------------------------------
    #  Weather – numeric
    # ------------------------------------------------------------------
    temperature_col = ColumnToText(
        "Temperature(F)",
        short_description="air temperature (°F)",
        value_map=lambda x: f"{x} °F" if pd.notna(x) else "missing",
    )

    wind_chill_col = ColumnToText(
        "Wind_Chill(F)",
        short_description="wind‑chill temperature (°F)",
        value_map=lambda x: f"{x} °F" if pd.notna(x) else "missing",
    )

    humidity_col = ColumnToText(
        "Humidity(%)",
        short_description="relative humidity (%)",
        value_map=lambda x: f"{x} %" if pd.notna(x) else "missing",
    )

    pressure_col = ColumnToText(
        "Pressure(in)",
        short_description="barometric pressure (inHg)",
        value_map=lambda x: f"{x} inHg" if pd.notna(x) else "missing",
    )

    visibility_col = ColumnToText(
        "Visibility(mi)",
        short_description="visibility (miles)",
        value_map=lambda x: f"{x} mi" if pd.notna(x) else "missing",
    )

    wind_speed_col = ColumnToText(
        "Wind_Speed(mph)",
        short_description="wind speed (mph)",
        value_map=lambda x: f"{x} mph" if pd.notna(x) else "missing",
    )

    precip_col = ColumnToText(
        "Precipitation(in)",
        short_description="precipitation (inches)",
        value_map=lambda x: f"{x} in" if pd.notna(x) else "missing",
    )

    # ------------------------------------------------------------------
    #  Weather – categorical
    # ------------------------------------------------------------------
    wind_dir_col = ColumnToText(
        "Wind_Direction",
        short_description="wind direction",
        value_map=lambda x: x if pd.notna(x) else "variable/unknown",
    )

    weather_cond_col = ColumnToText(
        "Weather_Condition", short_description="weather condition", value_map=lambda x: x if pd.notna(x) else "unknown"
    )

    timezone_col = ColumnToText(
        "Timezone", short_description="time‑zone of the location", value_map=lambda x: x if pd.notna(x) else "unknown"
    )

    # ------------------------------------------------------------------
    #  Boolean infrastructure flags
    # ------------------------------------------------------------------
    amenity_col = ColumnToText(
        "Amenity", short_description="nearby amenity present", value_map={True: "Yes", False: "No", pd.NA: "unknown"}
    )

    crossing_col = ColumnToText(
        "Crossing",
        short_description="roadway crossing present",
        value_map={True: "Yes", False: "No", pd.NA: "unknown"},
    )

    junction_col = ColumnToText(
        "Junction", short_description="road junction present", value_map={True: "Yes", False: "No", pd.NA: "unknown"}
    )

    station_col = ColumnToText(
        "Station", short_description="traffic station nearby", value_map={True: "Yes", False: "No", pd.NA: "unknown"}
    )

    stop_col = ColumnToText(
        "Stop", short_description="stop sign present", value_map={True: "Yes", False: "No", pd.NA: "unknown"}
    )

    traffic_signal_col = ColumnToText(
        "Traffic_Signal",
        short_description="traffic signal present",
        value_map={True: "Yes", False: "No", pd.NA: "unknown"},
    )

    # ------------------------------------------------------------------
    #  Twilight‑phase categorical variables
    # ------------------------------------------------------------------
    sunrise_sunset_col = ColumnToText(
        "Sunrise_Sunset",
        short_description="day‑time or night‑time",
        value_map={"Day": "day‑time", "Night": "night‑time", pd.NA: "unknown"},
    )

    civil_twilight_col = ColumnToText(
        "Civil_Twilight",
        short_description="civil twilight phase",
        value_map={"Day": "day", "Night": "night", pd.NA: "unknown"},
    )

    nautical_twilight_col = ColumnToText(
        "Nautical_Twilight",
        short_description="nautical twilight phase",
        value_map={"Day": "day", "Night": "night", pd.NA: "unknown"},
    )

    astronomical_twilight_col = ColumnToText(
        "Astronomical_Twilight",
        short_description="astronomical twilight phase",
        value_map={"Day": "day", "Night": "night", pd.NA: "unknown"},
    )

    severity_col = ColumnToText(
        "Severity",
        short_description="Severity level being 0 (minor / moderate impact) or 1 (significant / major impact)",
        value_map={0: "minor / moderate traffic impact", 1: "significant / major traffic impact"},
    )


class Reentry(Enum):
    reentry_numeric_qa = DirectNumericQA(
        column="Severity",
        text="On a scale being 0 (minor / moderate impact) or 1 (significant / major impact), what is the accident's severity level?",
    )

    reentry_qa = MultipleChoiceQA(
        column="Severity",
        text="How severe was the accident?",
        choices=(
            Choice("Level 1 or 2 - minor / moderate traffic impact", 0),
            Choice("Level 3 or 4 - significant / major traffic impact", 1),
        ),
    )


discretize_cols = [
    "Start_Time",
    "End_Time",
    "Start_Lat",
    "Start_Lng",
    "Distance(mi)",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
]
