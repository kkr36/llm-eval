"""
Plan: For each column in the dataset, randomly pick n. Out of these n columns, make these the outcome variables, discretize them, and get average results.

Problems:
- what if each feature has a distinct number of options - if dataset A had many more options than dataset B, don't we just trivially expect lower scores from dataset A?
"""

import pdb
from enum import Enum

import pandas as pd
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


data = pd.read_csv("../data/diabetes.csv")

OUTCOMES = ["readmitted"]

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short-term hospital",
    3: "Discharged/transferred to skilled nursing facility (SNF)",
    4: "Discharged/transferred to intermediate care facility (ICF)",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left against medical advice",
    8: "Discharged/transferred to home under care of a home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to a swing bed",
    16: "Discharged/transferred/referred to another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "Null (or not mapped)",
    19: "Expired at home",
    20: "Expired in a medical facility",
    21: "Expired, place unknown",
    22: "Discharged/transferred to a rehab facility including rehab units of a hospital",
    23: "Discharged/transferred to a long-term care hospital",
    24: "Discharged/transferred to a nursing facility certified as a Medicare swing bed",
    25: "Discharged/transferred to another rehab facility",
    26: "Discharged/transferred to a critical access hospital",
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred to a psychiatric hospital or psychiatric distinct part unit",
    29: "Discharged/transferred to a critical access hospital",
    30: "Discharged/transferred to another Type of Health Care Institution not defined elsewhere",
}

ADMISSION_SOURCE_MAP = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another healthcare facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal delivery",
    12: "Premature delivery",
    13: "Sick baby",
    14: "Extramural birth",
    15: "Transfer from another hospital for outpatient services",
    17: "Transfer from another healthcare facility for outpatient services",
    18: "Transfer from hospital inpatient",
    19: "Transfer from hospital outpatient",
    20: "Transfer from ambulatory surgery center",
    21: "Transfer from hospice",
    22: "Transfer from rehabilitation facility",
    23: "Transfer from long-term care hospital",
    24: "Transfer from psychiatric hospital or unit",
    25: "Transfer from intermediate care facility",
    #     26: "Transfer from residential care facility",
    #     27: "Transfer from ambulatory surgery center",
    #     28: "Transfer from rehabilitation facility including rehab units of a hospital",
    #     29: "Transfer from critical access hospital",
    #     30: "Transfer from federal health care facility",
    #     99: "Unknown/other"
}

PAYER_CODE_MAPPING = {
    "MC": "Medicare",
    "MD": "Medicaid",
    "BC": "Blue Cross",
    "SP": "Self-pay",
    "CM": "Champus",
    "UN": "United Healthcare",
    "DM": "Department of Defense",
    "CP": "Champus/Tricare",
    "PP": "Private Insurance",
    "WC": "Worker's Compensation",
    "HM": "HMO (Health Maintenance Organization)",
    "OG": "Other Government",
    "PO": "Other Private",
    "CH": "ChampVA",
    "MP": "Managed Care, Private",
    "OT": "Other",
    "SI": "Self-Insured",
    "FR": "Federal Government",
    "?": "Unknown",
}


def categorize_icd9(code):
    """
    Categorizes an ICD-9 diagnosis code into a broader category based on the UCI Diabetes dataset.

    :param code: str, ICD-9 diagnosis code (can be a number or an 'E'/'V' code)
    :return: str, category name
    """
    DIAG_MAPPING = {
        range(1, 140): "Infectious and parasitic diseases",
        range(140, 240): "Neoplasms (cancers)",
        range(240, 250): "Endocrine, nutritional, and metabolic diseases",
        "250": "Diabetes mellitus",
        range(251, 280): "Other endocrine disorders",
        range(280, 290): "Diseases of the blood and blood-forming organs",
        range(290, 320): "Mental disorders",
        range(320, 390): "Diseases of the nervous system and sense organs",
        range(390, 460): "Diseases of the circulatory system",
        range(460, 520): "Diseases of the respiratory system",
        range(520, 580): "Diseases of the digestive system",
        range(580, 630): "Diseases of the genitourinary system",
        range(630, 680): "Complications of pregnancy, childbirth, and the puerperium",
        range(680, 710): "Diseases of the skin and subcutaneous tissue",
        range(710, 740): "Diseases of the musculoskeletal system and connective tissue",
        range(740, 760): "Congenital anomalies",
        range(760, 780): "Certain conditions originating in the perinatal period",
        range(780, 800): "Symptoms, signs, and ill-defined conditions",
        range(800, 1000): "Injury and poisoning",
        "E": "External causes of injury and poisoning",
        "V": "Factors influencing health status and contact with health services",
        "?": "Unknown or missing",
    }

    if not code or code == "?":
        return DIAG_MAPPING["?"]

    code = str(code)

    # Handle E and V codes
    if code.startswith("E"):
        return DIAG_MAPPING["E"]
    elif code.startswith("V"):
        return DIAG_MAPPING["V"]

    # Try converting to an integer for range checking
    try:
        num_code = int(float(code))  # Convert decimal strings like '250.02' to 250
        for key_range, category in DIAG_MAPPING.items():
            if isinstance(key_range, range) and num_code in key_range:
                return category
        return "Unknown"
    except ValueError:
        return "Invalid Code"


class ColumnsEncoding(Enum):
    RACE_COL = ColumnToText(
        "race",
        short_description="race",
        value_map={x: x for x in set(data["race"].tolist())},
    )

    GENDER_COL = ColumnToText(
        "gender",
        short_description="gender",
        value_map={x: x for x in set(data["gender"].tolist())},
    )

    AGE_COL = ColumnToText(
        "age",
        short_description="age in years",
        value_map={x: f"{x} years" for x in set(data["age"].tolist())},
    )

    WEIGHT_COL = ColumnToText(
        "weight",
        short_description="weight in lbs",
        value_map={x: x for x in set(data["weight"].tolist())},
    )

    ADMISSION_ID = ColumnToText(
        "admission_type_id",
        short_description="method of admission",
        value_map={
            1: "Emergency",
            2: "Urgent",
            3: "Elective",
            4: "Newborn",
            5: "Not Available",
            6: "Null (or blank)",
            7: "Trauma Center",
            8: "Not Mapped",
        },
    )

    DISCHARGE_DISPOSITION_ID_COL = ColumnToText(
        "discharge_disposition_id",
        short_description="discharge information",
        value_map=DISCHARGE_DISPOSITION_MAP,
    )

    ADMISSION_SOURCE_COL = ColumnToText(
        "admission_source_id",
        short_description="admission circumstances",
        value_map=ADMISSION_SOURCE_MAP,
    )

    TIME_HOSP_COL = ColumnToText(
        "time_in_hospital",
        short_description="Integer number of days between admission and discharge",
        value_map=lambda x: f"{x} days",
    )

    PAYER_CODE_COL = ColumnToText(
        "payer_code",
        short_description="primary payer of expenses",
        value_map=PAYER_CODE_MAPPING,
    )

    MEDICAL_SPECIALTY_COL = ColumnToText(
        "medical_specialty",
        short_description="specialty of admitting physician",
        value_map={x: x for x in set(data["medical_specialty"].tolist())},
    )

    NUM_LAB_PROC_COL = ColumnToText(
        "num_lab_procedures",
        short_description="Number of lab tests performed during the encounter",
        value_map=lambda x: f"{x} tests",
    )

    NUM_PROC_COL = ColumnToText(
        "num_procedures",
        short_description="Number of procedures (other than lab tests) performed during the encounter",
        value_map={x: x for x in set(data["num_procedures"].tolist())},
    )

    MEDICATIONS_COL = ColumnToText(
        "num_medications",
        short_description="Number of distinct generic names administered during the encounter",
        value_map=lambda x: f"{x} medications",
    )

    OUTPATIENT_COL = ColumnToText(
        "number_outpatient",
        short_description="Number of outpatient visits of the patient in the year preceding the encounter",
        value_map=lambda x: f"{x} visits",
    )

    EMERGENCY_COL = ColumnToText(
        "number_emergency",
        short_description="Number of emergency visits of the patient in the year preceding the encounter",
        value_map=lambda x: f"{x} visits",
    )

    # emergency_col = ColumnToText(
    #     "number_inpatient",
    #     short_description="Number of inpatient visits of the patient in the year preceding the encounter",
    #     value_map=lambda x: f"{x} visits"
    # )

    DIAG_1_COL = ColumnToText(
        "diag_1",
        short_description="primary diagnosis",
        value_map=lambda x: categorize_icd9(
            x
        ),  # NOTE not using a dict is fine because we'll never turn this into an MC question
    )

    DIAG_2_COL = ColumnToText(
        "diag_2",
        short_description="secondary diagnosis",
        value_map=lambda x: categorize_icd9(x),
    )

    DIAG_3_COL = ColumnToText(
        "diag_3",
        short_description="tertiary diagnosis",
        value_map=lambda x: categorize_icd9(x),
    )

    NUMBER_DIAG_COL = ColumnToText(
        "number_diagnoses",
        short_description="Number of diagnoses entered to the system",
        value_map=lambda x: f"{x} diagnoses",
    )

    MAX_GLU_SERUM_COL = ColumnToText(
        "max_glu_serum",
        short_description="Max glucose serum",
        value_map={x: f"{x} mg / dL" for x in set(data["max_glu_serum"].tolist())},
    )

    A1C_COL = ColumnToText(
        "A1Cresult",
        short_description="A1C level",
        value_map={x: f"{x} %" for x in set(data["A1Cresult"].tolist())},
    )

    METFORMIN_COL = ColumnToText(
        "metformin",
        short_description="metformin dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    REPAGLINIDE_COL = ColumnToText(
        "repaglinide",
        short_description="repaglinide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    NATEGLINIDE_COL = ColumnToText(
        "nateglinide",
        short_description="nateglinide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    CHLORPROPAMIDE_COL = ColumnToText(
        "chlorpropamide",
        short_description="chlorpropamide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    GLIMEPIRIDE_COL = ColumnToText(
        "glimepiride",
        short_description="glimepiride dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    ACETOHEXAMIDE_COL = ColumnToText(
        "acetohexamide",
        short_description="acetohexamide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    GLIPIZIDE_COL = ColumnToText(
        "glipizide",
        short_description="glipizide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    TOLBUTAMIDE_COL = ColumnToText(
        "tolbutamide",
        short_description="tolbutamide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    PIOGLITAZONE_COL = ColumnToText(
        "pioglitazone",
        short_description="pioglitazone dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    ROSIGLITAZONE = ColumnToText(
        "rosiglitazone",
        short_description="rosiglitazone dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    ACARBOSE = ColumnToText(
        "acarbose",
        short_description="acarbose dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    MIGLITOL = ColumnToText(
        "miglitol",
        short_description="miglitol dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    TROGLITAZONE = ColumnToText(
        "troglitazone",
        short_description="troglitazone dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    TOLAZAMIDE = ColumnToText(
        "tolazamide",
        short_description="tolazamide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    EXAMIDE = ColumnToText(
        "examide",
        short_description="examide dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    CITOGLIPTON = ColumnToText(
        "citoglipton",
        short_description="citoglipton dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    INSULIN = ColumnToText(
        "insulin",
        short_description="insulin dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    GLYBURIDE_METFORMIN = ColumnToText(
        "glyburide-metformin",
        short_description="glyburide-metformin dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    GLIPIZIDE_METFORMIN = ColumnToText(
        "glipizide-metformin",
        short_description="glipizide-metformin dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    GLIMEPIRIDE_PIOGLITAZONE = ColumnToText(
        "glimepiride-pioglitazone",
        short_description="glimepiride-pioglitazone dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    METFORMIN_ROSIGLITAZONE = ColumnToText(
        "metformin-rosiglitazone",
        short_description="metformin-rosiglitazone dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    METFORMIN_PIOGLITAZONE = ColumnToText(
        "metformin-pioglitazone",
        short_description="metformin-pioglitazone dosage change during encounter",
        value_map={
            "Down": "went down",
            "Up": "went up",
            "Steady": "stayed steady",
            "No": "not prescribed",
        },
    )

    CHANGE = ColumnToText(
        "change",
        short_description="change in diabetic medications (either dosage or generic name)",
        value_map={"Ch": "Change", "No": "No change"},
    )

    DIABETESMED = ColumnToText(
        "diabetesMed",
        short_description="any diabetic medication prescribed",
        value_map={x: x for x in set(data["diabetesMed"].tolist())},
    )

    OUTCOME = ColumnToText(
        "readmitted",
        short_description="whether patient was readmitted with 30 days",
        value_map=lambda x: 1 if x == "<30" else 0,
    )


class Reentry(Enum):
    reentry_numeric_qa = DirectNumericQA(
        column="readmitted",
        text=("Was the patient readmitted within 30 days?"),
    )

    reentry_qa = MultipleChoiceQA(
        column="readmitted",
        text="Was the patient readmitted within 30 days?",
        choices=(
            Choice("Yes, they were", 1),
            Choice("No, they were not", 0),
        ),
    )


discretize_cols = [
    "time_in_hospital",
    "number_diagnoses",
    "num_lab_procedures",
    "num_medications",
]
