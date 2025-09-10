from string import Template


THE_QUESTION = "Is this mushroom poisonous or edible?"
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
The following data describes physical and environmental characteristics of different mushroom species. \
Each entry includes attributes such as cap shape, odor, gill size, spore print color, and more. \
The goal is to determine whether a mushroom is edible or poisonous based on these features. \
Please answer the following question using the information provided.
"""
