import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA

class ColumnsEncoding(Enum):

    # buying,maint,doors,persons,lug_boot,safety,acceptable
    buying_col = ColumnToText(
        "buying",
        short_description="Initial buying price of the car",
        value_map={
            "vhigh": "Very high buying price",
            "high": "High buying price",
            "med": "Medium buying price",
            "low": "Low buying price"
        }
    )

    maint_col = ColumnToText(
        "maint",
        short_description="Ongoing maintenance cost",
        value_map={
            "vhigh": "Very high maintenance cost",
            "high": "High maintenance cost",
            "med": "Medium maintenance cost",
            "low": "Low maintenance cost"
        }
    )

    doors_col = ColumnToText(
        "doors",
        short_description="Number of doors",
        value_map={
            "2": "2-door car",
            "3": "3-door car",
            "4": "4-door car",
            "5more": "Car with 5 or more doors"
        }
    )

    persons_col = ColumnToText(
        "persons",
        short_description="Passenger capacity",
        value_map={
            "2": "Seats 2 people",
            "4": "Seats 4 people",
            "more": "Seats more than 4 people"
        }
    )

    lug_boot_col = ColumnToText(
        "lug_boot",
        short_description="Size of the luggage boot",
        value_map={
            "small": "Small luggage capacity",
            "med": "Medium luggage capacity",
            "big": "Large luggage capacity"
        }
    )

    safety_col = ColumnToText(
        "safety",
        short_description="Estimated safety rating",
        value_map={
            "low": "Low safety rating",
            "med": "Moderate safety rating",
            "high": "High safety rating"
        }
    )

    acceptable_col = ColumnToText(
        "acceptable",
        short_description="Car acceptability",
        value_map={
            0: "not acceptable",
            1: "acceptable"
        }
    )


class Reentry(Enum):

    reentry_numeric_qa = DirectNumericQA(
        column='acceptable',
        text=(
            "Is the car acceptable?"
        ),
    )

    reentry_qa = MultipleChoiceQA(
        column='acceptable',
        text="Is the car acceptable?",
        choices=(
            Choice("Yes, the car is acceptable.", 1),
            Choice("No, the car is not acceptable.", 0),
        ),
    )



OUTCOMES = ["acceptable"]
discretize_cols = []