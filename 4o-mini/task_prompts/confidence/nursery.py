from string import Template


THE_QUESTION = "Was this application classified as priority?"
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
The following data corresponds to nursery school application evaluations. \
Each record includes social, financial, and personal information about a child's family situation. \
We are aiming to classify each application into two categories: as priority or not. \
Please answer each question based on the information provided. \
The data provided is sufficient to make an informed decision for each application.
"""