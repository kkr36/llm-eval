import pdb
from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


class ColumnsEncoding(Enum):
    """COLUMNS"""

    fLength = ColumnToText(
        "fLength",
        short_description="major axis of ellipse [mm]",
        value_map=lambda x: f"{x} mm",
    )

    fWidth = ColumnToText("fWidth", short_description="minor axis of ellipse [mm]", value_map=lambda x: f"{x} mm")

    fSize = ColumnToText(
        "fSize", short_description="10-log of sum of content of all pixels [in #phot]", value_map=lambda x: f"{x} phot"
    )

    fConc = ColumnToText(
        "fConc", short_description="ratio of sum of two highest pixels over fSize [ratio]", value_map=lambda x: f"{x}"
    )

    fConc1 = ColumnToText(
        "fConc1", short_description="ratio of highest pixel over fSize [ratio]", value_map=lambda x: f"{x}"
    )

    fAsym = ColumnToText(
        "fAsym",
        short_description="distance from highest pixel to center, projected onto major axis [mm]",
        value_map=lambda x: f"{x} mm",
    )

    fM3Long = ColumnToText(
        "fM3Long", short_description="3rd root of third moment along major axis [mm]", value_map=lambda x: f"{x} mm"
    )

    fM3Trans = ColumnToText(
        "fM3Trans", short_description="3rd root of third moment along minor axis [mm]", value_map=lambda x: f"{x} mm"
    )

    fAlpha = ColumnToText(
        "fAlpha", short_description="angle of major axis with vector to origin [deg]", value_map=lambda x: f"{x} deg"
    )

    fDist = ColumnToText(
        "fDist", short_description="distance from origin to center of ellipse [mm]", value_map=lambda x: f"{x} mm"
    )

    class_binary = ColumnToText(
        "class_binary", short_description="gamma (signal), hadron (background)", value_map={0: "g", 1: "h"}
    )


class Reentry(Enum):
    reentry_numeric_qa = DirectNumericQA(
        column="class_binary",
        text=("Is the observation caused by primary gammas (signal)?"),
    )

    reentry_qa = MultipleChoiceQA(
        column="class_binary",
        text="Is the observation caused by primary gammas (signal)?",
        choices=(
            Choice("Yes, it is", 0),
            Choice("No, it is hadronic (background)", 1),
        ),
    )


OUTCOMES = ["class_binary"]
discretize_cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
