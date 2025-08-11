import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA

OUTCOMES = ["facility"]

# TASK_DESCRIPTION = """\
# The following data corresponds to international health data on pregnant mothers who are residents of Latin countries. \
# The data is from states throughout Latin countries in the years 2005-2010. \
# Please answer each question based on the information provided. \
# The data provided is enough to reach an approximate answer for each person.
# """

class ColumnsEncoding(Enum):

    antenatal_col = ColumnToText(
        "antenatal_visits",
        short_description="number of antenatal visits during pregnancy",
        value_map=lambda x: f"{int(x)} antenatal visits",
    )

    birth_num_col = ColumnToText(
        "birth_num",
        short_description="number of prior births",
        value_map=lambda x: f"number {int(x)} child borne by the mother"
    )

    gender_col = ColumnToText(
        "sex",
        short_description="gender",
        value_map={
            1: "Male",
            2: "Female",
        },
    )

    past_term_col = ColumnToText(
        "past_terminate",
        short_description="whether mother has had prior terminated pregnancies",
        value_map={
            1: "yes",
            0: "no"
        }
    )

    healthcare_col = ColumnToText(
        "heathcare_visit",
        short_description="visited by a family planning worker in the past year",
        value_map={
            1: "yes",
            0: "no"
        }
    )

    doctor_col = ColumnToText(
        "doctor",
        short_description="visited a doctor during pregnancy",
        value_map={
            1: "yes",
            0: "no"
        }
    )

    nurse_col = ColumnToText(
        "nurse",
        short_description="visited a nurse during pregnancy",
        value_map={
            1: "yes",
            0: "no"
        }
    )

    aux_nurse_col = ColumnToText(
        "aux_nurse",
        short_description="visited a auxiliary nurse during pregnancy",
        value_map={
            1: "yes",
            0: "no"
        }
    )

    mother_education_col = ColumnToText(
        "mother_education",
        short_description="mother's education",
            value_map={
            0: 'has no formal education',
            1: 'attended up to primary school',
            2: 'attended up to secondary school',
            3: 'attended up to post-secondary school'
        }
    )

    mother_height_col = ColumnToText(
        "mother_height",
        short_description="height of mother (cm)",
        value_map=lambda x: f"{x/10} cm tall"
    )

    mother_weight_col = ColumnToText(
        "mother_weight",
        short_description="weight of mother (kg)",
        value_map=lambda x: f"{x/10} kg"
    )

    urban_col = ColumnToText(
        "urban",
        short_description="living environment",
        value_map={
            1: 'urban',
            2: 'rural'
        }
    )

    age_col = ColumnToText(
        "mother_age",
        short_description="age of mother (yrs)",
        value_map=lambda x: f"{int(x)} years old",
    )

    facility_col = ColumnToText(
        "facility",
        short_description="whether child was born inside medical facility",
        value_map={
            1: "yes",
            0: "no"
        }
    )

    country_col = ColumnToText(
        "Country",
        short_description="country of residence of mother",
        value_map={
            "PE": "Peru",
            "BO": "Bolivia",
            "GY": "Guyana",
            "HN": "Honduras",
            "CO": "Colombia",
            "HT": "Haiti"
        }
    )

class Reentry(Enum):

    reentry_numeric_qa = DirectNumericQA(
        column='facility',
        text=(
            "Was the child born in a medical facility?"
        ),
    )


    reentry_qa = MultipleChoiceQA(
        column='facility',
        text="Was the child born in a medical facility?",
        choices=(
            Choice("Yes, they were", 1),
            Choice("No, they were not", 0),
        ),
    )

discretize_cols = [ "antenatal_visits", "birth_num", "mother_height", "mother_weight", "mother_age"]
