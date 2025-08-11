import pdb
from tqdm import tqdm
from enum import Enum
from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import MultipleChoiceQA, Choice, DirectNumericQA

class ColumnsEncoding(Enum):

    clump_thickness = ColumnToText(
        "Clump_thickness",
        short_description="Clump_thickness on a scale from 1-10",
        value_map=lambda x: f"{x} from 1-10"
    )

    uniformity_size = ColumnToText(
        "Uniformity_of_cell_size",
        short_description="Uniformity_of_cell_size on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    uniformity_shape = ColumnToText(
        "Uniformity_of_cell_shape",
        short_description="Uniformity_of_cell_shape on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    adhesion = ColumnToText(
        "Marginal_adhesion",
        short_description="Marginal_adhesion on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    single_epithelial = ColumnToText(
        "Single_epithelial_cell_size",
        short_description="Single_epithelial_cell_size on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    bare_nuclei = ColumnToText(
        "Bare_nuclei",
        short_description="Bare_nuclei on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    bland = ColumnToText(
        "Bland_chromatin",
        short_description="Bland_chromatin on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    normal = ColumnToText(
        "Normal_nucleoli",
        short_description="Normal_nucleoli on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    mitoses = ColumnToText(
        "Mitoses",
        short_description="Mitoses on a scale from 1-10",
    value_map=lambda x: f"{x} from 1-10"
    )

    class_col = ColumnToText(
        "Class",
        short_description="Class",
        value_map=lambda x: {0: 'benign', 1:'malignant'}[x],
    )

class Reentry(Enum):

    reentry_numeric_qa = None

    reentry_qa = MultipleChoiceQA(
        column='Class',
        text="Is this sample benign or malignant?",
        choices=(
            Choice("Benign", 0),
            Choice("Malignant", 1),
        ),
    )


OUTCOMES = ['Class']

discretize_cols = ['Clump_thickness','Uniformity_of_cell_size', 'Uniformity_of_cell_shape', 'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei', 'Normal_nucleoli', 'Mitoses']

