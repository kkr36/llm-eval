from string import Template


THE_QUESTION = "Is this car considered acceptable?"
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
The following data corresponds to car evaluation records derived from a hierarchical decision-making model. \
Each record describes characteristics related to the car's price and technical features, including comfort and safety. \
The goal is to classify each car as either acceptable or not acceptable for use. \
Please answer each question based on the features provided. \
The data is comprehensive enough to support a well-informed classification.
"""