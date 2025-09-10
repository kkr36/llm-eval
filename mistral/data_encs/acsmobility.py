from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA

from data_encs.acsencodings import Encodings


OUTCOMES = ["MIG"]


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

    GENDER_COL = ColumnToText(
        "SEX",
        short_description="gender",
        value_map={
            1: "Male",
            2: "Female",
        },
    )

    disability_col = ColumnToText(
        "DIS", short_description="disability status", value_map={1: "With a disability", 2: "Without a disability"}
    )

    parent_col = ColumnToText(
        "ESP",
        short_description="employment status of parents",
        value_map={
            8: "Living with mother: Mother not in labor force",
            0: "N/A (not own child of householder, and not child in subfamily)",
            2: "Living with two parents: Father only in labor force",
            5: "Living with father: Father in the labor force",
            4: "Living with two parents: Neither parent in labor force",
            7: "Living with mother: Mother in the labor force",
            6: "Living with father: Father not in labor force",
            1: "Living with two parents: Both parents in labor force",
            3: "Living with two parents: Mother only in labor force",
        },
    )

    citizen_col = ColumnToText(
        "CIT",
        short_description="citizenship status",
        value_map={
            3: "Born abroad of American parent(s)",
            2: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
            4: "U.S. citizen by naturalization",
            1: "Born in the U.S.",
            5: "Not a citizen of the U.S.",
        },
    )

    military_col = ColumnToText(
        "MIL",
        short_description="military service",
        value_map={
            0: "N/A (less than 17 years old)",
            1: "Now on active duty",
            4: "Never served in the military",
            2: "On active duty in the past, but not now",
            3: "Only on active duty for training in Reserves/National Guard",
        },
    )

    ancestry_col = ColumnToText(
        "ANC",
        short_description="number ancestries reported",
        value_map={
            4: "Not reported",
            8: "Suppressed for data year 2018 for select PUMAs",
            3: "Unclassified",
            2: "Multiple",
            1: "Single",
        },
    )

    nativity_col = ColumnToText(
        "NATIVITY", short_description="nativity for US", value_map={2: "Foreign born", 1: "Native"}
    )

    RELP_COL = ColumnToText(
        "RELP",
        short_description="relationship to head of household",
        value_map={k: v + " of head of household" for k, v in (Encodings.RELP_DICT.value).items()},
    )

    hearing_col = ColumnToText("DEAR", short_description="hearing difficulty", value_map={2: "No", 1: "Yes"})

    eye_col = ColumnToText("DEYE", short_description="vision difficulty", value_map={2: "No", 1: "Yes"})

    cognitive_col = ColumnToText(
        "DREM",
        short_description="cognitive difficulty",
        value_map={
            2: "No",
            1: "Yes",
            0: "N/A (less than 5 years old)",
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

    GCL_COL = ColumnToText(
        "GCL",
        short_description="Grandparents living with grandchildren",
        value_map={0: "N/A (less than 30 years/institutional GQ)", 1: "Yes", 2: "No"},
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

    ESR_COL = ColumnToText(
        "ESR",
        short_description="employment status",
        value_map={
            5: "Armed Forces, With a Job But Not At Work",
            3: "Unemployed",
            2: "Civilian employed, with a job but not at work",
            1: "Civilian employed, at work",
            6: "Not in Labor Force",
            0: "N/A (less than 16 years old)",
            4: "Armed Forces, At Work",
        },
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

    JWMNP_COL = ColumnToText(
        "JWMNP", short_description="travel time to work in minutes", value_map=lambda x: f"{x} minutes"
    )

    INCOME_COL = ColumnToText("PINCP", short_description="annual income in US dollars", value_map=lambda x: f"${x}")

    MIG_COL = ColumnToText(
        "MIG",
        short_description="Mobility status (lived here 1 year ago)",
        value_map={1: "Yes, same house", 0: "No, lived somewhere else 1 year ago"},
    )


class Reentry(Enum):
    reentry_qa = MultipleChoiceQA(
        column="MIG",
        text="Did this person have the same residential address 1 year ago?",
        choices=(
            Choice("Yes, they lived in the same house", 1),
            Choice("No, they lived in a different house 1 year ago", 0),
        ),
    )

    reentry_numeric_qa = DirectNumericQA(
        column="MIG",
        text=("Did this person have the same residential address 1 year ago?"),
    )


discretize_cols = ["AGEP", "WKHP", "JWMNP", "PINCP"]
