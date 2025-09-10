import pdb
from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


OUTCOMES = ["is_Cammeo"]


class ColumnsEncoding(Enum):
    area_col = ColumnToText(
        "Area",
        short_description="the number of pixels within the boundaries of the rice grain",
        value_map=lambda x: f"{x} pixels",
    )

    perimeter_col = ColumnToText(
        "Perimeter",
        short_description="total length around the rice grain boundary",
        value_map=lambda x: f"{x:.2f} units",
    )

    major_axis_col = ColumnToText(
        "Major_Axis_Length",
        short_description="length of the longest axis of the rice grain",
        value_map=lambda x: f"{x:.2f} units",
    )

    minor_axis_col = ColumnToText(
        "Minor_Axis_Length",
        short_description="length of the shortest axis of the rice grain",
        value_map=lambda x: f"{x:.2f} units",
    )

    eccentricity_col = ColumnToText(
        "Eccentricity",
        short_description="ratio describing the elongation of the rice grain",
        value_map=lambda x: f"{x:.4f}",
    )

    convex_area_col = ColumnToText(
        "Convex_Area",
        short_description="number of pixels in the convex hull of the rice grain",
        value_map=lambda x: f"{x} pixels",
    )

    extent_col = ColumnToText(
        "Extent",
        short_description="ratio of area to bounding box area of the rice grain",
        value_map=lambda x: f"{x:.4f}",
    )

    Cammeo_col = ColumnToText(
        "is_Cammeo", short_description="label: is Cammeo (1) or Osmancik (0)", value_map={0: "Osmancik", 1: "Cammeo"}
    )


class Reentry(Enum):
    reentry_numeric_qa = DirectNumericQA(
        column="is_Cammeo",
        text=("Is the rice grain Cammeo rather than Osmancik?"),
    )

    reentry_qa = MultipleChoiceQA(
        column="is_Cammeo",
        text="Is the rice grain Cammeo rather than Osmancik?",
        choices=(
            Choice("Yes, the rice grain is Cammeo", 1),
            Choice("No, the rice grain is Osmancik", 0),
        ),
    )


discretize_cols = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]
