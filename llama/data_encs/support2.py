import pdb
from enum import Enum

import pandas as pd
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


class ColumnsEncoding(Enum):
    """COLUMNS"""

    def num_to_text(unit=""):
        unit = f" {unit}" if unit else ""
        return lambda x: f"{x}{unit}" if pd.notna(x) else "missing"

    age_col = ColumnToText("age", "age in years", value_map=num_to_text("yr"))
    slos_col = ColumnToText("slos", "days from study entry to discharge", value_map=num_to_text("d"))
    dtime_col = ColumnToText("d.time", "days of follow-up", value_map=num_to_text("d"))
    numco_col = ColumnToText("num.co", "number of comorbidities", value_map=num_to_text())
    edu_col = ColumnToText("edu", "years of education", value_map=num_to_text("yr"))
    scoma_col = ColumnToText("scoma", "coma score day 3", value_map=num_to_text())
    charges_col = ColumnToText("charges", "hospital charges", value_map=num_to_text("USD"))
    totcst_col = ColumnToText("totcst", "total RCC cost", value_map=num_to_text("USD"))
    totmcst_col = ColumnToText("totmcst", "total micro cost", value_map=num_to_text("USD"))
    avtisst_col = ColumnToText("avtisst", "average TISS score days 3-25", value_map=num_to_text())
    sps_col = ColumnToText("sps", "SUPPORT physiology score day 3", value_map=num_to_text())
    aps_col = ColumnToText("aps", "APACHE III physiology score day 3", value_map=num_to_text())
    surv2m_col = ColumnToText("surv2m", "model 2-month survival estimate", value_map=num_to_text())
    surv6m_col = ColumnToText("surv6m", "model 6-month survival estimate", value_map=num_to_text())
    hday_col = ColumnToText("hday", "hospital day at study entry", value_map=num_to_text())
    prg2m_col = ColumnToText("prg2m", "physician 2-month survival estimate", value_map=num_to_text())
    prg6m_col = ColumnToText("prg6m", "physician 6-month survival estimate", value_map=num_to_text())
    dnrday_col = ColumnToText("dnrday", "day of DNR order", value_map=num_to_text())
    meanbp_col = ColumnToText("meanbp", "mean arterial blood pressure", value_map=num_to_text("mmHg"))
    wblc_col = ColumnToText("wblc", "white blood cells (thousands)", value_map=num_to_text("k"))
    hrt_col = ColumnToText("hrt", "heart rate", value_map=num_to_text("bpm"))
    resp_col = ColumnToText("resp", "respiration rate", value_map=num_to_text("breaths/min"))
    temp_col = ColumnToText("temp", "body temperature", value_map=num_to_text("C"))
    pafi_col = ColumnToText("pafi", "PaO2/FiO2 ratio", value_map=num_to_text())
    alb_col = ColumnToText("alb", "serum albumin", value_map=num_to_text("g/dL"))
    bili_col = ColumnToText("bili", "bilirubin", value_map=num_to_text("mg/dL"))
    crea_col = ColumnToText("crea", "creatinine", value_map=num_to_text("mg/dL"))
    sod_col = ColumnToText("sod", "serum sodium", value_map=num_to_text("mEq/L"))
    ph_col = ColumnToText("ph", "arterial blood pH", value_map=num_to_text())
    glucose_col = ColumnToText("glucose", "glucose", value_map=num_to_text("mg/dL"))
    bun_col = ColumnToText("bun", "blood urea nitrogen", value_map=num_to_text("mg/dL"))
    urine_col = ColumnToText("urine", "urine output", value_map=num_to_text("mL"))
    adlp_col = ColumnToText("adlp", "ADL index (patient)", value_map=num_to_text())
    adls_col = ColumnToText("adls", "ADL index (surrogate)", value_map=num_to_text())
    adlsc_col = ColumnToText("adlsc", "imputed ADL calibrated", value_map=num_to_text())

    diabetes_col = ColumnToText("diabetes", "diabetes comorbidity", value_map={1: "Yes", 0: "No", pd.NA: "unknown"})
    dementia_col = ColumnToText("dementia", "dementia comorbidity", value_map={1: "Yes", 0: "No", pd.NA: "unknown"})
    hospdead_col = ColumnToText("hospdead", "death in hospital", value_map={1: "Yes", 0: "No", pd.NA: "unknown"})
    # death_col    = ColumnToText("death",    "death before 31‑Dec‑1994",
    #                             value_map={1: "Yes", 0: "No", pd.NA: "unknown"})

    sex_col = ColumnToText("sex", "sex", value_map=lambda x: x if pd.notna(x) else "unknown")
    dzgroup_col = ColumnToText("dzgroup", "disease subcategory", value_map=lambda x: x if pd.notna(x) else "unknown")
    dzclass_col = ColumnToText("dzclass", "disease category", value_map=lambda x: x if pd.notna(x) else "unknown")
    income_col = ColumnToText("income", "income bracket", value_map=lambda x: x if pd.notna(x) else "unknown")
    race_col = ColumnToText("race", "race", value_map=lambda x: x if pd.notna(x) else "unknown")
    ca_col = ColumnToText("ca", "cancer status", value_map=lambda x: x if pd.notna(x) else "unknown")
    dnr_col = ColumnToText("dnr", "DNR order status", value_map=lambda x: x if pd.notna(x) else "unknown")
    sfdm2_col = ColumnToText("sfdm2", "functional disability", value_map=lambda x: x if pd.notna(x) else "missing")


class Reentry(Enum):
    """QUESTIONS"""

    reentry_numeric_qa = DirectNumericQA(
        column="hospdead", text="Did the patient die in the hospital? Answer 1 for yes, 0 for no."
    )

    reentry_qa = MultipleChoiceQA(
        column="hospdead",
        text="Did the patient die in the hospital?",
        choices=(
            Choice("Yes, the patient died in hospital", 1),
            Choice("No, the patient survived to discharge", 0),
        ),
    )


OUTCOMES = ["hospdead"]
discretize_cols = [
    "age",
    "slos",
    "d.time",
    "num.co",
    "edu",
    "scoma",
    "charges",
    "totcst",
    "totmcst",
    "avtisst",
    "sps",
    "aps",
    "surv2m",
    "surv6m",
    "hday",
    "prg2m",
    "prg6m",
    "dnrday",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
    "adlp",
    "adls",
    "adlsc",
]
