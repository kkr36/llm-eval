from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA

from data_encs.acsencodings import Encodings


OUTCOMES = ["PINCP"]


class ColumnsEncoding(Enum):
    AGE_COL = ColumnToText(
        "AGEP",
        short_description="age",
        value_map=lambda x: f"{int(x)} years old",
    )

    EDUCATION_COL = ColumnToText(
        "SCHL",
        short_description="education",
        value_map={
            4: "Grade 1",
            1: "No schooling completed",
            7: "Grade 4",
            16: "Regular high school diploma",
            3: "Kindergarten",
            23: "Professional degree beyond a bachelor's degree",
            19: "1 or more years of college credit, no degree",
            10: "Grade 7",
            22: "Master's degree",
            20: "Associate's degree",
            0: "N/A (less than 3 years old)",
            2: "Nursery school, preschool",
            21: "Bachelor's degree",
            8: "Grade 5",
            24: "Doctorate degree",
            14: "Grade 11",
            6: "Grade 3",
            17: "GED or alternative credential",
            12: "Grade 9",
            13: "Grade 10",
            9: "Grade 6",
            5: "Grade 2",
            15: "12th grade - no diploma",
            11: "Grade 8",
            18: "Some college, but less than 1 year",
        },
    )

    RACE_COL = ColumnToText(
        "RAC1P",
        short_description="race",
        value_map={
            3: "American Indian alone",
            1: "White alone",
            8: "Some Other Race alone",
            6: "Asian alone",
            9: "Two or More Races",
            2: "Black or African American alone",
            4: "Alaska Native alone",
            7: "Native Hawaiian and Other Pacific Islander alone",
            5: "American Indian and Alaska Native tribes; or not specified and no other races",
        },
    )

    GENDER_COL = ColumnToText(
        "SEX",
        short_description="gender",
        value_map={
            1: "Male",
            2: "Female",
        },
    )

    COW_COL = ColumnToText(
        "COW",
        short_description="class of worker",
        value_map={
            9: "Unemployed and last worked 5 years ago or earlier or never worked",
            1: "Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions",
            7: "Self-employed in own incorporated business, professional practice or farm",
            8: "Working without pay in family business or farm",
            3: "Local government employee (city, county, etc.)",
            6: "Self-employed in own not incorporated business, professional practice, or farm",
            4: "State government employee",
            5: "Federal government employee",
            2: "Employee of a private not-for-profit, tax-exempt, or charitable organization",
            0: "Not in universe (less than 16 years old/NILF who last worked more than 5 years ago or never worked)",
        },
    )

    MAR_COL = ColumnToText(
        "MAR",
        short_description="marriage status",
        value_map={
            1: "Married",
            5: "Never married or under 15 years old",
            4: "Separated",
            3: "Divorced",
            2: "Widowed",
        },
    )

    OCCP_COL = ColumnToText(
        "OCCP",
        short_description="job type",
        value_map={k: "works as " + v for k, v in (Encodings.OCCP_DICT.value).items()},
    )

    POBP_COL = ColumnToText(
        "POBP",
        short_description="place of birth",
        value_map={k: "born in " + v for k, v in (Encodings.POBP_DICT.value).items()},
    )

    RELP_COL = ColumnToText(
        "RELP",
        short_description="relationship to head of household",
        value_map={k: v + " of head of household" for k, v in (Encodings.RELP_DICT.value).items()},
    )

    WKHP_COL = ColumnToText(
        "WKHP",
        short_description="hours worked per week",
        value_map=lambda x: f"works {x} hours per week",
    )

    STATE_COL = ColumnToText(
        "STATE",
        short_description="state of residence",
        value_map=lambda x: f"resident of {x}",
    )

    INCOME_COL = ColumnToText(
        "PINCP",
        short_description="Whether individual earns more than $50000 annually",
        value_map={1: "yes", 0: "no"},
    )


class Reentry(Enum):
    reentry_qa = MultipleChoiceQA(
        column="PINCP",
        text="Does this person earn more than $50000 annually?",
        choices=(
            Choice("Yes, they earn more than $50000 annually", 1),
            Choice("No, they do not earn more than $50000 annually", 0),
        ),
    )

    reentry_numeric_qa = DirectNumericQA(
        column="PINCP",
        text=("Does this person earn more than $50000 annually?"),
    )


discretize_cols = ["AGEP"]
