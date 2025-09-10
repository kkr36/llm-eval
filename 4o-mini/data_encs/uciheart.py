import pdb
from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


attributes_45 = variable_mappings = {
    "age": "Age in years",
    "sex": {"description": "Sex", "values": {0: "female", 1: "male"}},
    "cp": {
        "description": "Chest pain type",
        "values": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    },
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": {"description": "Fasting blood sugar > 120 mg/dl", "values": {0: "false", 1: "true"}},
    "restecg": {
        "description": "Resting electrocardiographic results",
        "values": {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
    },
    "thalach": "Maximum heart rate achieved",
    "exang": {"description": "Exercise-induced angina", "values": {0: "no", 1: "yes"}},
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": {
        "description": "Slope of peak exercise ST segment",
        "values": {1: "upsloping", 2: "flat", 3: "downsloping"},
    },
    "ca": "Number of major vessels colored by fluoroscopy (0–3)",
    "thal": {"description": "Thalassemia status", "values": {3: "normal", 6: "fixed defect", 7: "reversible defect"}},
    "num": {
        "description": "Diagnosis of heart disease",
        "values": {0: "< 50% diameter narrowing", 1: "> 50% diameter narrowing"},
    },
}

testing_data = 45
attributes_dict = {45: attributes_45}


class ColumnsEncoding(Enum):
    age = ColumnToText("age", short_description="age", value_map=lambda x: x)

    sex = ColumnToText(
        "sex",
        short_description="sex",
        value_map=lambda x: attributes_dict[testing_data]["sex"]["values"][x],
    )

    cp = ColumnToText(
        "cp",
        short_description="constrictive pericarditis",
        value_map=lambda x: attributes_dict[testing_data]["cp"]["values"][x],
    )

    trestbps = ColumnToText(
        "trestbps", short_description="resting blood pressure (on hospital admission)", value_map=lambda x: x
    )

    chol = ColumnToText("chol", short_description="serum cholestoral", value_map=lambda x: x)

    fbs = ColumnToText(
        "fbs",
        short_description="fasting blood sugar > 120 mg/dl",
        value_map=lambda x: attributes_dict[testing_data]["fbs"]["values"][x],
    )

    restecg = ColumnToText(
        "restecg",
        short_description="resting ECG",
        value_map=lambda x: attributes_dict[testing_data]["restecg"]["values"][x],
    )

    thalach = ColumnToText("thalach", short_description="maximum heart rate achieved (bpm)", value_map=lambda x: x)

    exang = ColumnToText(
        "exang",
        short_description="exercise induced angina",
        value_map=lambda x: attributes_dict[testing_data]["exang"]["values"][x],
    )

    oldpeak = ColumnToText(
        "oldpeak", short_description="ST depression induced by exercise relative to rest (mm)", value_map=lambda x: x
    )

    slope = ColumnToText(
        "slope",
        short_description="ST segment/heart rate (ST/HR) slope",
        value_map=lambda x: attributes_dict[testing_data]["slope"]["values"][x],
    )

    ca = ColumnToText("ca", short_description="# major vessels colored by flourosopy (0-3)", value_map=lambda x: x)

    thal = ColumnToText(
        "thal",
        short_description="thal",
        value_map=lambda x: attributes_dict[testing_data]["thal"]["values"][x],
    )

    num = ColumnToText("num", short_description="diagnosis of heart disease", value_map=lambda x: x)


class Reentry(Enum):
    reentry_numeric_qa = None

    reentry_qa = MultipleChoiceQA(
        column="num",
        text="Does this person have heart disease?",
        choices=(
            Choice("No heart disease", 0),
            Choice("Yes, heart disease", 1),
        ),
    )


OUTCOMES = ["num"]

discretize_cols = ["age", "trestbps", "chol", "thalach", "ca"]
