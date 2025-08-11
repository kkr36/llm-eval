from string import Template


THE_QUESTION = "Is the grade of this glioma low-grade (LGG) or a glioblastoma (GBM)?"
TASK_TEMPLATE = Template(
    """Provide your best guess and the probability that it is correct (0.0 to 1.0) for
        the following question. Give ONLY the guess and probability, no other words or
        explanation. For example:\n\nGuess: <most likely guess, as short as possible; not
        a complete sentence, just the guess!>\n Probability: <the probability between 0.0
        and 1.0 that your guess is correct, without any extra commentary whatsoever; just
        the probability!>\n\nThe question is: $the_question"""
)

TASK_DESCRIPTION = TASK_TEMPLATE.substitute(the_question=THE_QUESTION)

CONTEXT = """\
The following data corresponds to clinical and genetic features of patients diagnosed with glioma. \
This includes patient demographics (such as age, gender, and race) as well as the mutation status of various genes commonly associated with glioma. \
Each entry describes a unique patient. \
Please answer the following question based on the information provided. \
The data is sufficient to determine whether the patient has a low-grade glioma (LGG) or glioblastoma (GBM).
"""