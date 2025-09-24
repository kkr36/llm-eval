from string import Template


THE_QUESTION = "Is the median housing value greater than $200,000?"
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
The following dataset contains housing information from California census blocks from the 1990 Census. \
In this sample a block group on average includes 1425.5 individuals living in a geographically compact area. \
Each record includes demographic, geographic, and housing-related attributes. \
The task is to classify whether the median house value in a neighborhood in California, 1990 greater than 200,000, \
Use the provided attributes to make an informed classification.
"""
