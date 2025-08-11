from string import Template


THE_QUESTION = "Is the individual unemployed?"

TASK_TEMPLATE = Template(
    """Provide your best guess and the probability that it is correct (0.0 to 1.0) for
        the following question. Give ONLY the guess and probability, no other words or
        explanation. For example:\n\nGuess: <most likely guess, as short as possible; not
        a complete sentence, just the guess!>\n Probability: <the probability between 0.0
        and 1.0 that your guess is correct, without any extra commentary whatsoever; just
        the probability!>\n\nThe question is: $the_question"""
)

TASK_DESCRIPTION = TASK_TEMPLATE.substitute(the_question=THE_QUESTION)

CONTEXT = """The following data corresponds to US Census data on US residents. \
The data is from states throughout the US in the year 2018. Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person."""

