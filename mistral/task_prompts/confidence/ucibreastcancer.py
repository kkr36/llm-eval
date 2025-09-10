from string import Template


THE_QUESTION = "Is this sample malignant or benign?"
TASK_TEMPLATE = Template(
    """Provide your best guess and the probability that it is correct (0.0 to 1.0) for
        the following question. Give ONLY the guess and probability, no other words or
        explanation. For example:\n\nGuess: <most likely guess, as short as possible; not
        a complete sentence, just the guess!>\n Probability: <the probability between 0.0
        and 1.0 that your guess is correct, without any extra commentary whatsoever; just
        the probability!>\n\nThe question is: $the_question"""
)

TASK_DESCRIPTION = TASK_TEMPLATE.substitute(the_question=THE_QUESTION)

CONTEXT = """Samples arrive periodically as a Wisconson doctor reports his clinical cases related to breast cancer from 1992. 
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person."""
