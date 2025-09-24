from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA

from data_encs.acsencodings import Encodings


OUTCOMES = ["JWMNP"]


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

    mig_col = ColumnToText(
        "MIG",
        short_description="Mobility status (lived here 1 year ago)",
        value_map={
            0: "N/A (less than 1 year old)",
            1: "Yes, same house (nonmovers)",
            2: "No, outside US and Puerto Rico",
            3: "No, different house in US or Puerto Rico",
        },
    )

    RELP_COL = ColumnToText(
        "RELP",
        short_description="relationship to head of household",
        value_map={k: v + " of head of household" for k, v in (Encodings.RELP_DICT.value).items()},
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

    STATE_COL = ColumnToText(
        "STATE",
        short_description="state of residence",
        value_map=lambda x: f"resident of {x}",
    )

    PUMA_COL = ColumnToText(
        "PUMA",
        short_description="Public use microdata area code (PUMA) based on 2010 Census definition",
        value_map=lambda x: x,
    )

    OCCP_COL = ColumnToText(
        "OCCP",
        short_description="job type",
        value_map={k: "works as " + v for k, v in (Encodings.OCCP_DICT.value).items()},
    )

    JWTR_COL = ColumnToText(
        "JWTR",
        short_description="Means of transportation to work",
        value_map={
            3: "Streetcar or trolley car (carro publico in Puerto Rico)",
            5: "Railroad",
            10: "Walked",
            6: "Ferryboat",
            1: "Car, Truck, Or Van",
            7: "Taxicab",
            0: "N/A (not a worker--not in the labor force, including persons under 16 years; unemployed; employed, with a job but not at",
            9: "Bicycle",
            11: "Worked At Home",
            12: "Other Method",
            8: "Motorcycle",
            2: "Bus or Trolley Bus",
            4: "Subway Or Elevated",
        },
    )

    POWPUMA_COL = ColumnToText(
        "POWPUMA",
        short_description="Place of work PUMA based on 2010 Census definitions",
        value_map=lambda x: "N/A (not a worker-not in the labor force, including persons under 16 years; unemployed; civilian employed, with a job not at work; Armed Forces, with a job but not at work)"
        if x == 0
        else "Did not work in the United States or in Puerto Rico"
        if x == 1
        else x,
    )

    POVPIP_COL = ColumnToText("POVPIP", short_description="Income-to-poverty ratio recode", value_map=lambda x: x)

    JWMNP_COL = ColumnToText(
        "JWMNP",
        short_description="Whether individual's travel time to work is longer than 20 minutes",
        value_map={1: "Yes", 0: "No"},
    )


class Reentry(Enum):
    reentry_qa = MultipleChoiceQA(
        column="JWMNP",
        text="Is this person's travel time to work longer than 20 minutes?",
        choices=(
            Choice("Yes", 1),
            Choice("No", 0),
        ),
    )

    reentry_numeric_qa = DirectNumericQA(
        column="JWMNP",
        text=("Is this person's travel time to work longer than 20 minutes?"),
    )


discretize_cols = ["AGEP", "POVPIP"]
