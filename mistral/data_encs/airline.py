import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA
import os

OUTCOMES = ["Delay"]

class ColumnsEncoding(Enum):

    time = ColumnToText(
        "Time",
        short_description="number of minutes departure is past 12:00 am",
        value_map=lambda x: x
    )

    dayofweek = ColumnToText(
        "DayOfWeek",
        short_description="day of week",
        value_map=lambda x: x
    )

    airportto = ColumnToText(
        "AirportTo",
        short_description="airport of arrival",
        value_map=lambda x: x
    )

    airportfrom = ColumnToText(
        "AirportFrom",
        short_description="airport of departure",
        value_map=lambda x: x
    )

    flight = ColumnToText(
        "Flight",
        short_description="flight number",
        value_map=lambda x: x
    )

    airline = ColumnToText(
        "Airline",
        short_description="airline name",
        value_map=lambda x: x
    )

    delay = ColumnToText(
        "Delay",
        short_description="whether flight was delayed",
        value_map={
            1: "yes",
            0: "no"
        },
    )

class Reentry(Enum):

    reentry_numeric_qa = DirectNumericQA(
        column='Delay',
        text=(
            "Did this flight get delayed?"
        ),
    )


    reentry_qa = MultipleChoiceQA(
        column='Delay',
        text="Did this flight get delayed?",
        choices=(
            Choice("Yes, it got delayed", 1),
            Choice("No, it did not get delayed", 0),
        ),
    )

discretize_cols = ['DayOfWeek', 'Time']



