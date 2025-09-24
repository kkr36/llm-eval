from string import Template


THE_QUESTION = "Does the patient have diabetes?"

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
This dataset contains diagnostic health information for female patients of Pima Indian heritage, all at least 21 years old. \
Each record includes medical predictor variables such as glucose level, BMI, age, insulin level, and more. \
The goal is to classify whether a patient has diabetes or not, based on these medical indicators. \
Use the data provided to make an informed diagnosis prediction.
"""
