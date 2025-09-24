import pdb
from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


OUTCOMES = ["Outcome"]


class ColumnsEncoding(Enum):
    pregnancies_col = ColumnToText(
        "Pregnancies",
        short_description="Number of pregnancies the patient has had",
        value_map=lambda x: x,  # Numeric feature
    )

    glucose_col = ColumnToText("Glucose", short_description="Plasma glucose concentration", value_map=lambda x: x)

    blood_pressure_col = ColumnToText(
        "BloodPressure", short_description="Diastolic blood pressure (mm Hg)", value_map=lambda x: x
    )

    skin_thickness_col = ColumnToText(
        "SkinThickness", short_description="Triceps skin fold thickness (mm)", value_map=lambda x: x
    )

    insulin_col = ColumnToText("Insulin", short_description="2-hour serum insulin (mu U/ml)", value_map=lambda x: x)

    bmi_col = ColumnToText(
        "BMI", short_description="Body Mass Index (weight in kg / height in m²)", value_map=lambda x: x
    )

    diabetes_pedigree_col = ColumnToText(
        "DiabetesPedigreeFunction",
        short_description="Diabetes pedigree function (likelihood of diabetes based on family history)",
        value_map=lambda x: x,
    )

    age_col = ColumnToText("Age", short_description="Age of the patient (years)", value_map=lambda x: x)

    diabetes_col = ColumnToText("Outcome", short_description="diabetes diagnosis", value_map={1: "Yes", 0: "No"})


class Reentry(Enum):
    reentry_numeric_qa = DirectNumericQA(
        column="Outcome",
        text=("Does this female patient have diabetes?"),
    )

    reentry_qa = MultipleChoiceQA(
        column="Outcome",
        text="Does this female patient have diabetes?",
        choices=(
            Choice("Yes, she has diabetes", 1),
            Choice("No, she doesn't have diabetes", 0),
        ),
    )


discretize_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
