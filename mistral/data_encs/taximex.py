import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA
from datetime import datetime

class ColumnsEncoding(Enum):

    pickup_datetime_col = ColumnToText(
        "pickup_datetime",
        short_description="pickup date and time",
        value_map=lambda x: f"pickup occurred on {(datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).strftime('%A, %B %d, %Y at %I:%M %p')}"
    )

    pickup_longitude_col = ColumnToText(
        "pickup_longitude",
        short_description="pickup longitude",
        value_map=lambda x: f"pickup longitude at {x:.5f}"
    )

    pickup_latitude_col = ColumnToText(
        "pickup_latitude",
        short_description="pickup latitude",
        value_map=lambda x: f"pickup latitude at {x:.5f}"
    )

    dropoff_longitude_col = ColumnToText(
        "dropoff_longitude",
        short_description="dropoff longitude",
        value_map=lambda x: f"dropoff longitude at {x:.5f}"
    )

    dropoff_latitude_col = ColumnToText(
        "dropoff_latitude",
        short_description="dropoff latitude",
        value_map=lambda x: f"dropoff latitude at {x:.5f}"
    )

    dist_meters_col = ColumnToText(
        "dist_meters",
        short_description="trip distance in meters",
        value_map=lambda x: f"trip covered {x} meters"
    )

    wait_sec_col = ColumnToText(
        "wait_sec",
        short_description="waiting time in seconds",
        value_map=lambda x: f"waited {x} seconds before pickup"
    )

    month_col = ColumnToText(
        "month",
        short_description="month of pickup",
        value_map=lambda x: f"pickup occurred in month {x}"
    )

    week_col = ColumnToText(
        "week",
        short_description="week number of pickup",
        value_map=lambda x: f"pickup occurred in week {x}"
    )

    weekday_col = ColumnToText(
        "weekday",
        short_description="day of the week",
        value_map=lambda x: f"pickup occurred on weekday {x}"
    )

    hour_col = ColumnToText(
        "hour",
        short_description="hour of pickup",
        value_map=lambda x: f"pickup occurred at hour {x}"
    )

    minute_oftheday_col = ColumnToText(
        "minute_oftheday",
        short_description="minute of the day",
        value_map=lambda x: f"pickup occurred at minute {x} of the day"
    )

    trip_duration_col = ColumnToText(
        "trip_duration",
        short_description="trip duration exceed 30 minutes (1800 seconds)",
        value_map={0: 'No', 1: 'Yes'}
    )

class Reentry(Enum):

    reentry_numeric_qa = DirectNumericQA(
        column='trip_duration',
        text=(
            "Does the total ride duration time exceed 30 minutes (1800 seconds)? Answer 1 for yes, 0 for no."
        ),
    )

    reentry_qa = MultipleChoiceQA(
        column='trip_duration',
        text="Does the total ride duration time exceed 30 minutes (1800 seconds)?",
        choices=(
            Choice("Yes, the total ride duration time exceeds 30 minutes (1800 seconds)", 1),
            Choice("No, the total ride duration time does not exceed 30 minutes (1800 seconds)", 0),
        ),
    )

OUTCOMES = ["trip_duration"]
discretize_cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
                   'dist_meters', 'wait_sec', 'month', 'week', 'weekday', 'hour', 'minute_oftheday']