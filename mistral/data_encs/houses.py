import pdb
from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


class ColumnsEncoding(Enum):
    median_house_value_col = ColumnToText(
        "median_house_value",
        short_description="Median house value in the neighborhood in California, 1990",
        value_map={0: "not greater than 200,000", 1: "greater than 200,000"},
    )

    median_income_col = ColumnToText(
        "median_income",
        short_description="Median household income in the neighborhood in California, 1990",
        value_map=lambda x: x,
    )

    housing_median_age_col = ColumnToText(
        "housing_median_age",
        short_description="Median age of the houses in the neighborhood in California, 1990",
        value_map=lambda x: x,
    )

    total_rooms_col = ColumnToText(
        "total_rooms", short_description="Total number of rooms in the block", value_map=lambda x: x
    )

    total_bedrooms_col = ColumnToText(
        "total_bedrooms", short_description="Total number of bedrooms in the block", value_map=lambda x: x
    )

    population_col = ColumnToText(
        "population", short_description="Total population in the block", value_map=lambda x: x
    )

    households_col = ColumnToText(
        "households", short_description="Number of households in the block", value_map=lambda x: x
    )

    latitude_col = ColumnToText(
        "latitude", short_description="Geographical latitude of the block", value_map=lambda x: x
    )

    longitude_col = ColumnToText(
        "longitude", short_description="Geographical longitude of the block", value_map=lambda x: x
    )


class Reentry(Enum):
    reentry_numeric_qa = DirectNumericQA(
        column="median_house_value",
        text=("Is the median housing value greater than 200,000?"),
    )

    reentry_qa = MultipleChoiceQA(
        column="median_house_value",
        text="Is the housing value greater than 200,000?",
        choices=(
            Choice("Yes, the housing value is greater than 200,000.", 1),
            Choice("No, the housing value is not greater than 200,000.", 0),
        ),
    )


OUTCOMES = ["median_house_value"]
discretize_cols = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "latitude",
    "longitude",
]
