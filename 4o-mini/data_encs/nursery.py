import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA

class ColumnsEncoding(Enum):

    parents_col = ColumnToText(
        "parents",
        short_description="Quality of parents",
        value_map={
            "usual": "Usual parental care",
            "pretentious": "Pretentious parental care",
            "great_pret": "Highly pretentious parental care"
        }
    )

    has_nurs_col = ColumnToText(
        "has_nurs",
        short_description="Nursery service availability",
        value_map={
            "proper": "Proper nursery care",
            "less_proper": "Less proper nursery care",
            "improper": "Improper nursery care",
            "critical": "Critical nursery need",
            "very_crit": "Very critical nursery need"
        }
    )

    form_col = ColumnToText(
        "form",
        short_description="Completion of formal application",
        value_map={
            "complete": "Fully complete application",
            "completed": "Recently completed application",
            "incomplete": "Incomplete application",
            "foster": "Foster care situation"
        }
    )

    children_col = ColumnToText(
        "children",
        short_description="Number of children",
        value_map={
            "1": "1 child",
            "2": "2 children",
            "3": "3 children",
            "more": "More than 3 children"
        }
    )

    housing_col = ColumnToText(
        "housing",
        short_description="Housing situation",
        value_map={
            "convenient": "Convenient housing",
            "less_conv": "Less convenient housing",
            "critical": "Critical housing condition"
        }
    )

    finance_col = ColumnToText(
        "finance",
        short_description="Financial situation",
        value_map={
            "convenient": "Financially convenient",
            "inconv": "Financially inconvenient"
        }
    )

    social_col = ColumnToText(
        "social",
        short_description="Social conditions",
        value_map={
            "nonprob": "No social problems",
            "slightly_prob": "Slight social problems",
            "problematic": "Problematic social conditions"
        }
    )

    health_col = ColumnToText(
        "health",
        short_description="Health status",
        value_map={
            "recommended": "Recommended health status",
            "priority": "Priority health case",
            "not_recom": "Not recommended health status"
        }
    )

    acceptable_col = ColumnToText(
        "recommendation_priority",
        short_description="admission recommendation priority",
        value_map={
            0: "not priority",
            1: "priority"
        }
    )

class Reentry(Enum):

    reentry_numeric_qa = DirectNumericQA(
        column='recommendation_priority',
        text=(
            "Is this application classified as priority?"
        ),
    )

    reentry_qa = MultipleChoiceQA(
        column='recommendation_priority',
        text="Is this application classified as priority?",
        choices=(
            Choice("Yes, it's classified as priority.", 1),
            Choice("No, it's not classified as priority.", 0),
        ),
    )


OUTCOMES = ["recommendation_priority"]
discretize_cols = []
