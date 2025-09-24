import pdb
from enum import Enum

import pandas as pd
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


OUTCOMES = ["y_binary"]
data = pd.read_csv("../data/bank.csv")


"""COLUMNS"""


class ColumnsEncoding(Enum):
    age = ColumnToText(
        "age",
        short_description="years of age",
        value_map=lambda x: f"{x} years",
    )

    job = ColumnToText(
        "job",
        short_description="job type",
        value_map=lambda x: f"{x}",
    )

    marital = ColumnToText(
        "marital",
        short_description="marital status",
        value_map=lambda x: f"{x}",
    )

    education = ColumnToText(
        "education",
        short_description="highest level of education",
        value_map=lambda x: f"{x}",
    )

    default = ColumnToText(
        "default",
        short_description="history of prior defaults",
        value_map=lambda x: f"{x}",
    )

    balance = ColumnToText(
        "balance",
        short_description="average yearly balance (euros)",
        value_map=lambda x: f"{x} euros",
    )

    housing = ColumnToText(
        "housing",
        short_description="existing of housing loan",
        value_map=lambda x: f"{x}",
    )

    loan = ColumnToText(
        "loan",
        short_description="existing of personal loan",
        value_map=lambda x: f"{x}",
    )

    contact = ColumnToText(
        "contact",
        short_description="contact communication type for last contact of the current campaign",
        value_map=lambda x: f"{x}",
    )

    day = ColumnToText(
        "day",
        short_description="last contact day of the month",
        value_map=lambda x: f"{x} {'st' if x == 1 else 'nd' if x == 2 else 'rd' if x == 3 else 'th'} day of the month",
    )

    month = ColumnToText("month", short_description="last contact month of year", value_map=lambda x: x)

    duration = ColumnToText(
        "duration", short_description="last contact duration, in seconds", value_map=lambda x: f"{x} s"
    )

    campaign = ColumnToText(
        "campaign",
        short_description="number of contacts performed during this campaign and for this client",
        value_map=lambda x: f"{x} contacts",
    )

    pdays = ColumnToText(
        "pdays",
        short_description="number of days that passed by after the client was last contacted from a previous campaign",
        value_map={x: x if x != -1 else "client not previously contacted" for x in set(data["pdays"])},
    )

    previous = ColumnToText(
        "previous",
        short_description="number of contacts performed before this campaign and for this client",
        value_map=lambda x: x,
    )

    poutcome = ColumnToText(
        "poutcome", short_description="outcome of the previous marketing campaign", value_map=lambda x: x
    )

    y = ColumnToText(
        "y_binary", short_description="has the client subscribed to a term deposit?", value_map={0: "no", 1: "yes"}
    )


reentry_numeric_qa = DirectNumericQA(
    column="y_binary",
    text=("Has the client subscribed a term deposit?"),
)


class Reentry(Enum):
    reentry_numeric_qa = DirectNumericQA(
        column="y_binary",
        text=("Has the client subscribed a term deposit?"),
    )

    reentry_qa = MultipleChoiceQA(
        column="y_binary",
        text="Has the client subscribed a term deposit?",
        choices=(
            Choice("Yes, they have", 1),
            Choice("No, they have not", 0),
        ),
    )


discretize_cols = ["age, balance, day"]
