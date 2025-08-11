import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA

discretize_cols = ['Age_at_diagnosis']  

# Target variable: Glioma Grade
grade_dict = {
    0: "low-grade glioma (LGG)",
    1: "glioblastoma (GBM)"
}

# Gender
gender_dict = {
    0: "male",
    1: "female"
}

# Race
race_dict = {
    0: "White",
    1: "Black or African American",
    2: "Asian",
    3: "American Indian or Alaska Native"
}

# Gene mutation status
mutation_dict = {
    0: "NOT_MUTATED",
    1: "MUTATED"
}

class ColumnsEncoding(Enum):
    # Grade (target)
    grade_col = ColumnToText(
        "Grade",
        short_description="glioma grade",
        value_map={k: f"The patient has a {v}" for k, v in grade_dict.items()}
    )

    # Gender
    gender_col = ColumnToText(
        "Gender",
        short_description="gender",
        value_map={k: f"The patient is {v}" for k, v in gender_dict.items()}
    )

    # Age at diagnosis
    age_col = ColumnToText(
        "Age_at_diagnosis",
        short_description="age at diagnosis",
        value_map=lambda x: f"The patient was diagnosed at {float(x)} years old"
    )

    # Race
    race_col = ColumnToText(
        "Race",
        short_description="race",
        value_map={k: f"The patient's race is {v}" for k, v in race_dict.items()}
    )


    IDH1_col = ColumnToText(
        "IDH1",
        short_description="IDH1 mutation status",
        value_map={
            0: "IDH1 (isocitrate dehydrogenase (NADP(+)) 1) is NOT_MUTATED",
            1: "IDH1 (isocitrate dehydrogenase (NADP(+)) 1) is MUTATED"
        }
    )

    TP53_col = ColumnToText(
        "TP53",
        short_description="TP53 mutation status",
        value_map={
            0: "TP53 (tumor protein p53) is NOT_MUTATED",
            1: "TP53 (tumor protein p53) is MUTATED"
        }
    )

    ATRX_col = ColumnToText(
        "ATRX",
        short_description="ATRX mutation status",
        value_map={
            0: "ATRX (ATRX chromatin remodeler) is NOT_MUTATED",
            1: "ATRX (ATRX chromatin remodeler) is MUTATED"
        }
    )

    PTEN_col = ColumnToText(
        "PTEN",
        short_description="PTEN mutation status",
        value_map={
            0: "PTEN (phosphatase and tensin homolog) is NOT_MUTATED",
            1: "PTEN (phosphatase and tensin homolog) is MUTATED"
        }
    )

    EGFR_col = ColumnToText(
        "EGFR",
        short_description="EGFR mutation status",
        value_map={
            0: "EGFR (epidermal growth factor receptor) is NOT_MUTATED",
            1: "EGFR (epidermal growth factor receptor) is MUTATED"
        }
    )

    CIC_col = ColumnToText(
        "CIC",
        short_description="CIC mutation status",
        value_map={
            0: "CIC (capicua transcriptional repressor) is NOT_MUTATED",
            1: "CIC (capicua transcriptional repressor) is MUTATED"
        }
    )

    MUC16_col = ColumnToText(
        "MUC16",
        short_description="MUC16 mutation status",
        value_map={
            0: "MUC16 (mucin 16, cell surface associated) is NOT_MUTATED",
            1: "MUC16 (mucin 16, cell surface associated) is MUTATED"
        }
    )

    PIK3CA_col = ColumnToText(
        "PIK3CA",
        short_description="PIK3CA mutation status",
        value_map={
            0: "PIK3CA (phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha) is NOT_MUTATED",
            1: "PIK3CA (phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha) is MUTATED"
        }
    )

    NF1_col = ColumnToText(
        "NF1",
        short_description="NF1 mutation status",
        value_map={
            0: "NF1 (neurofibromin 1) is NOT_MUTATED",
            1: "NF1 (neurofibromin 1) is MUTATED"
        }
    )

    PIK3R1_col = ColumnToText(
        "PIK3R1",
        short_description="PIK3R1 mutation status",
        value_map={
            0: "PIK3R1 (phosphoinositide-3-kinase regulatory subunit 1) is NOT_MUTATED",
            1: "PIK3R1 (phosphoinositide-3-kinase regulatory subunit 1) is MUTATED"
        }
    )

    FUBP1_col = ColumnToText(
        "FUBP1",
        short_description="FUBP1 mutation status",
        value_map={
            0: "FUBP1 (far upstream element binding protein 1) is NOT_MUTATED",
            1: "FUBP1 (far upstream element binding protein 1) is MUTATED"
        }
    )

    RB1_col = ColumnToText(
        "RB1",
        short_description="RB1 mutation status",
        value_map={
            0: "RB1 (RB transcriptional corepressor 1) is NOT_MUTATED",
            1: "RB1 (RB transcriptional corepressor 1) is MUTATED"
        }
    )

    NOTCH1_col = ColumnToText(
        "NOTCH1",
        short_description="NOTCH1 mutation status",
        value_map={
            0: "NOTCH1 (notch receptor 1) is NOT_MUTATED",
            1: "NOTCH1 (notch receptor 1) is MUTATED"
        }
    )

    BCOR_col = ColumnToText(
        "BCOR",
        short_description="BCOR mutation status",
        value_map={
            0: "BCOR (BCL6 corepressor) is NOT_MUTATED",
            1: "BCOR (BCL6 corepressor) is MUTATED"
        }
    )

    CSMD3_col = ColumnToText(
        "CSMD3",
        short_description="CSMD3 mutation status",
        value_map={
            0: "CSMD3 (CUB and Sushi multiple domains 3) is NOT_MUTATED",
            1: "CSMD3 (CUB and Sushi multiple domains 3) is MUTATED"
        }
    )

    SMARCA4_col = ColumnToText(
        "SMARCA4",
        short_description="SMARCA4 mutation status",
        value_map={
            0: "SMARCA4 (SWI/SNF related, matrix associated, actin dependent regulator of chromatin, subfamily a, member 4) is NOT_MUTATED",
            1: "SMARCA4 (SWI/SNF related, matrix associated, actin dependent regulator of chromatin, subfamily a, member 4) is MUTATED"
        }
    )

    GRIN2A_col = ColumnToText(
        "GRIN2A",
        short_description="GRIN2A mutation status",
        value_map={
            0: "GRIN2A (glutamate ionotropic receptor NMDA type subunit 2A) is NOT_MUTATED",
            1: "GRIN2A (glutamate ionotropic receptor NMDA type subunit 2A) is MUTATED"
        }
    )

    IDH2_col = ColumnToText(
        "IDH2",
        short_description="IDH2 mutation status",
        value_map={
            0: "IDH2 (isocitrate dehydrogenase (NADP(+)) 2) is NOT_MUTATED",
            1: "IDH2 (isocitrate dehydrogenase (NADP(+)) 2) is MUTATED"
        }
    )

    FAT4_col = ColumnToText(
        "FAT4",
        short_description="FAT4 mutation status",
        value_map={
            0: "FAT4 (FAT atypical cadherin 4) is NOT_MUTATED",
            1: "FAT4 (FAT atypical cadherin 4) is MUTATED"
        }
    )

    PDGFRA_col = ColumnToText(
        "PDGFRA",
        short_description="PDGFRA mutation status",
        value_map={
            0: "PDGFRA (platelet-derived growth factor receptor alpha) is NOT_MUTATED",
            1: "PDGFRA (platelet-derived growth factor receptor alpha) is MUTATED"
        }
    )

class Reentry(Enum):

    reentry_numeric_qa = DirectNumericQA(
        column="Grade",
        text="What is the grade of this glioma?"
    )

    reentry_qa = MultipleChoiceQA(
        column="Grade",
        text="What is the grade of this glioma?",
        choices=(
            Choice("The patient has a low-grade glioma (LGG)", 0),
            Choice("The patient has a glioblastoma (GBM)", 1),
        ),
    )


OUTCOMES = ["Grade"]

