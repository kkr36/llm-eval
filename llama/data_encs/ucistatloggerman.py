import pdb
from enum import Enum

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from tqdm import tqdm


attributes_144 = {
    "Attribute1": {
        "type": "qualitative",
        "name": "Status of existing checking account",
        "values": {
            "A11": "< 0 DM",
            "A12": "0 <= ... < 200 DM",
            "A13": ">= 200 DM / salary assignments for at least 1 year",
            "A14": "no checking account",
        },
    },
    "Attribute3": {
        "type": "qualitative",
        "name": "Credit history",
        "values": {
            "A30": "no credits taken/ all credits paid back duly",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account/ other credits existing (not at this bank)",
        },
    },
    "Attribute4": {
        "type": "qualitative",
        "name": "Purpose",
        "values": {
            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation (does not exist?)",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
        },
    },
    "Attribute5": {"type": "numerical", "name": "Credit amount"},
    "Attribute6": {
        "type": "qualitative",
        "name": "Savings account/bonds",
        "values": {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "unknown/ no savings account",
        },
    },
    "Attribute7": {
        "type": "qualitative",
        "name": "Present employment since",
        "values": {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years",
        },
    },
    "Attribute8": {"type": "numerical", "name": "Installment rate in percentage of disposable income"},
    "Attribute9": {
        "type": "qualitative",
        "name": "Personal status and sex",
        "values": {
            "A91": "male: divorced/separated",
            "A92": "female: divorced/separated/married",
            "A93": "male: single",
            "A94": "male: married/widowed",
            "A95": "female: single",
        },
    },
    "Attribute10": {
        "type": "qualitative",
        "name": "Other debtors / guarantors",
        "values": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    },
    "Attribute11": {"type": "numerical", "name": "Present residence since"},
    "Attribute12": {
        "type": "qualitative",
        "name": "Property",
        "values": {
            "A121": "real estate",
            "A122": "building society savings agreement/life insurance",
            "A123": "car or other, not in attribute 6",
            "A124": "unknown / no property",
        },
    },
    "Attribute13": {"type": "numerical", "name": "Age in years"},
    "Attribute14": {
        "type": "qualitative",
        "name": "Other installment plans",
        "values": {"A141": "bank", "A142": "stores", "A143": "none"},
    },
    "Attribute15": {
        "type": "qualitative",
        "name": "Housing",
        "values": {"A151": "rent", "A152": "own", "A153": "for free"},
    },
    "Attribute16": {"type": "numerical", "name": "Number of existing credits at this bank"},
    "Attribute17": {
        "type": "qualitative",
        "name": "Job",
        "values": {
            "A171": "unemployed/unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/self-employed/highly qualified employee/officer",
        },
    },
    "Attribute19": {
        "type": "qualitative",
        "name": "Telephone",
        "values": {"A191": "none", "A192": "yes, registered under the customer's name"},
    },
    "Attribute20": {"type": "qualitative", "name": "Foreign worker", "values": {"A201": "yes", "A202": "no"}},
    "class": {
        "type": "qualitative",
        "name": "Credit Risk",
        "values": {
            0: "Good",
            1: "Bad",
        },
    },
}

attributes_dict = {144: attributes_144}
testing_data = 144


class ColumnsEncoding(Enum):
    a1 = ColumnToText(
        "Attribute1",
        short_description="Status of existing checking account",
        value_map=lambda x: attributes_dict[testing_data]["Attribute1"]["values"][x],
    )

    a2 = ColumnToText(
        "Attribute2",
        short_description="Duration of checking account in months",
        value_map=lambda x: f"{x}",
    )

    a3 = ColumnToText(
        "Attribute3",
        short_description="Credit history",
        value_map=lambda x: attributes_dict[testing_data]["Attribute3"]["values"][x],
    )

    a4 = ColumnToText(
        "Attribute4",
        short_description="Purpose of account",
        value_map=lambda x: attributes_dict[testing_data]["Attribute4"]["values"][x],
    )

    a5 = ColumnToText(
        "Attribute5",
        short_description="Credit amount (Germany 1994)",
        value_map=lambda x: f"{x}",
    )

    a6 = ColumnToText(
        "Attribute6",
        short_description="Savings account/bonds",
        value_map=lambda x: attributes_dict[testing_data]["Attribute6"]["values"][x],
    )

    a7 = ColumnToText(
        "Attribute7",
        short_description="Present employment since",
        value_map=lambda x: attributes_dict[testing_data]["Attribute7"]["values"][x],
    )

    a8 = ColumnToText(
        "Attribute8",
        short_description="Installment rate in percentage of disposable income",
        value_map=lambda x: f"{x}%",
    )

    a9 = ColumnToText(
        "Attribute9",
        short_description="Personal status and sex",
        value_map=lambda x: attributes_dict[testing_data]["Attribute9"]["values"][x],
    )

    a10 = ColumnToText(
        "Attribute10",
        short_description="Other debtors / guarantors",
        value_map=lambda x: attributes_dict[testing_data]["Attribute10"]["values"][x],
    )

    a11 = ColumnToText(
        "Attribute11",
        short_description="Present residence since",
        value_map=lambda x: f"{x}",
    )

    a12 = ColumnToText(
        "Attribute12",
        short_description="Property",
        value_map=lambda x: attributes_dict[testing_data]["Attribute12"]["values"][x],
    )

    a13 = ColumnToText(
        "Attribute13",
        short_description="Age",
        value_map=lambda x: f"{x}",
    )

    a14 = ColumnToText(
        "Attribute14",
        short_description="Other installment plans",
        value_map=lambda x: attributes_dict[testing_data]["Attribute14"]["values"][x],
    )

    a15 = ColumnToText(
        "Attribute15",
        short_description="Housing status",
        value_map=lambda x: attributes_dict[testing_data]["Attribute15"]["values"][x],
    )

    a16 = ColumnToText(
        "Attribute16",
        short_description="Number of existing credits at this bank",
        value_map=lambda x: f"{x}",
    )

    a17 = ColumnToText(
        "Attribute17",
        short_description="Job status",
        value_map=lambda x: attributes_dict[testing_data]["Attribute17"]["values"][x],
    )

    a18 = ColumnToText(
        "Attribute18",
        short_description="Number of people being liable to provide maintenance for",
        value_map=lambda x: f"{x} people",
    )

    a19 = ColumnToText(
        "Attribute19",
        short_description="Telephone",
        value_map=lambda x: attributes_dict[testing_data]["Attribute19"]["values"][x],
    )

    a20 = ColumnToText(
        "Attribute20",
        short_description="foreign worker",
        value_map=lambda x: attributes_dict[testing_data]["Attribute20"]["values"][x],
    )

    Class = ColumnToText(
        "class",
        short_description="credit score risks",
        value_map=lambda x: attributes_dict[testing_data]["class"]["values"][x],
    )


class Reentry(Enum):
    reentry_numeric_qa = None

    reentry_qa = MultipleChoiceQA(
        column="class",
        text="Does this person have a good credit or bad credit risk?",
        choices=(
            Choice("Yes, they have good credit", 0),
            Choice("No, they do not have good credit, they have bad credit", 1),
        ),
    )


OUTCOMES = ["class"]

discretize_cols = [
    "Attribute18",
    "Attribute16",
    "Attribute13",
    "Attribute11",
    "Attribute8",
    "Attribute5",
    "Attribute2",
]
