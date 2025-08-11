import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA

attributes_891 = variable_mappings = {
    "Diabetes_binary": {
        0: "No diabetes",
        1: "Prediabetes or diabetes"
    },
    "HighBP": {
        0: "No high blood pressure",
        1: "High blood pressure"
    },
    "HighChol": {
        0: "No high cholesterol",
        1: "High cholesterol"
    },
    "CholCheck": {
        0: "No cholesterol check in 5 years",
        1: "Had cholesterol check in 5 years"
    },
    "Smoker": {
        0: "Non-smoker (<100 cigarettes lifetime)",
        1: "Smoker (≥100 cigarettes lifetime)"
    },
    "Stroke": {
        0: "No stroke",
        1: "Had a stroke"
    },
    "HeartDiseaseorAttack": {
        0: "No CHD or heart attack",
        1: "CHD or heart attack"
    },
    "PhysActivity": {
        0: "No physical activity in past 30 days",
        1: "Had physical activity"
    },
    "Fruits": {
        0: "Consumes fruit <1 time/day",
        1: "Consumes fruit ≥1 time/day"
    },
    "Veggies": {
        0: "Consumes vegetables <1 time/day",
        1: "Consumes vegetables ≥1 time/day"
    },
    "HvyAlcoholConsump": {
        0: "Not a heavy drinker",
        1: "Heavy drinker"
    },
    "AnyHealthcare": {
        0: "No health coverage",
        1: "Has health coverage"
    },
    "NoDocbcCost": {
        0: "Could afford doctor",
        1: "Could not afford doctor"
    },
    "GenHlth": {
        1: "Excellent",
        2: "Very good",
        3: "Good",
        4: "Fair",
        5: "Poor"
    },
    "DiffWalk": {
        0: "No difficulty walking/climbing stairs",
        1: "Difficulty walking/climbing stairs"
    },
    "Sex": {
        0: "Female",
        1: "Male"
    },
    "Age": {
        1: "18-24",
        2: "25-29",
        3: "30-34",
        4: "35-39",
        5: "40-44",
        6: "45-49",
        7: "50-54",
        8: "55-59",
        9: "60-64",
        10: "65-69",
        11: "70-74",
        12: "75-79",
        13: "80 or older"
    },
    "Education": {
        1: "Never attended school or only kindergarten",
        2: "Grades 1-8 (Elementary)",
        3: "Grades 9-11 (Some high school)",
        4: "Grade 12 or GED (High school graduate)",
        5: "College 1-3 years (Some college or technical school)",
        6: "College 4+ years (College graduate)"
    },
    "Income": {
        1: "Less than $10,000",
        2: "$10,000 to < $15,000",
        3: "$15,000 to < $20,000",
        4: "$20,000 to < $25,000",
        5: "$25,000 to < $35,000",
        6: "$35,000 to < $50,000",
        7: "$50,000 to < $75,000",
        8: "$75,000 or more"
    }}
testing_data = 891
attributes_dict = {891:attributes_891}

class ColumnsEncoding(Enum):

    diabetes_binary = ColumnToText(
        "Diabetes_binary",
        short_description="Diabetes presence",
        value_map=lambda x: attributes_dict[testing_data]["Diabetes_binary"][x],
    )

    highbp = ColumnToText(
        "HighBP",
        short_description="Blood Pressure",
        value_map=lambda x: attributes_dict[testing_data]["HighBP"][x],
    )

    highchol = ColumnToText(
        "HighChol",
        short_description="high cholesterol",
        value_map=lambda x: attributes_dict[testing_data]["HighChol"][x],
    )

    cholcheck = ColumnToText(
        "CholCheck",
        short_description="cholesterol",
        value_map=lambda x: attributes_dict[testing_data]["CholCheck"][x],
    )

    bmi = ColumnToText(
        "BMI",
        short_description="Body Mass Index",
    value_map=lambda x: x
    )

    smoker = ColumnToText(
        "Smoker",
        short_description="Cigarette Smoker (>100 smoked)",
        value_map=lambda x: attributes_dict[testing_data]["Smoker"][x],
    )

    stroke = ColumnToText(
        "Stroke",
        short_description="Stroke",
        value_map=lambda x: attributes_dict[testing_data]["Stroke"][x],
    )

    heartdisease = ColumnToText(
        "HeartDiseaseorAttack",
        short_description="coronary heart disease/myocardial infarction",
        value_map=lambda x: attributes_dict[testing_data]["HeartDiseaseorAttack"][x],
    )

    physact = ColumnToText(
        "PhysActivity",
        short_description="physical activity past 30 days not including job",
        value_map=lambda x: attributes_dict[testing_data]["PhysActivity"][x],
    )

    fruits = ColumnToText(
        "Fruits",
        short_description="Consume Fruit >1/day",
        value_map=lambda x: attributes_dict[testing_data]["Fruits"][x],
    )

    veggies = ColumnToText(
        "Veggies",
        short_description="Consume Vegetables >1/day",
        value_map=lambda x: attributes_dict[testing_data]["Veggies"][x],
    )

    hvyalc = ColumnToText(
        "HvyAlcoholConsump",
        short_description="Heavy drinkers (adult men > 14/week and adult women > 7/week)",
        value_map=lambda x: attributes_dict[testing_data]["HvyAlcoholConsump"][x],
    )

    anyhealth = ColumnToText(
        "AnyHealthcare",
        short_description="Have any kind of health care coverage",
        value_map=lambda x: attributes_dict[testing_data]["AnyHealthcare"][x],
    )

    nodoc = ColumnToText(
        "NoDocbcCost",
        short_description="if in last year needed to see a doctor but could not because of cost",
        value_map=lambda x: attributes_dict[testing_data]["NoDocbcCost"][x],
    )

    genhlth = ColumnToText(
        "GenHlth",
        short_description="General health condition",
    value_map=lambda x: x
    )

    menhlth = ColumnToText(
        "MentHlth",
        short_description="# days during the past 30 days mental health was not good",
    value_map=lambda x: x
    )

    physhlth = ColumnToText(
        "PhysHlth",
        short_description="# days during the past 30 days physical health not good",
    value_map=lambda x: x
    )

    diffwalk = ColumnToText(
        "DiffWalk",
        short_description="difficulty walking/climbing stairs",
        value_map=lambda x: attributes_dict[testing_data]["DiffWalk"][x],
    )

    sex = ColumnToText(
        "Sex",
        short_description="sex",
        value_map=lambda x: attributes_dict[testing_data]["Sex"][x],
    )

    age = ColumnToText(
        "Age",
        short_description="13-level age category",
    value_map=lambda x: x
    )

    edu = ColumnToText(
        "Education",
        short_description="Education level ",
    value_map=lambda x: x
    )

    inc = ColumnToText(
        "Income",
        short_description="Income level:",
    value_map=lambda x: x
    )


class Reentry(Enum):

    reentry_numeric_qa = None

    reentry_qa = MultipleChoiceQA(
        column='Diabetes_binary',
        text="Does this person have diabetes or prediabetes?",
        choices=(
            Choice("No diabetes or prediabetes", 0),
            Choice("Yes, prediabetes or diabetes", 1),
        ),
    )


OUTCOMES = ['Diabetes_binary']

discretize_cols = ['BMI']
