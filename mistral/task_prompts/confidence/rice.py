from string import Template


THE_QUESTION = "Is the rice grain Cammeo rather than Osmancik?"
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
The following data corresponds to rice grain's classification. \
Rice grain's images were taken for the two species, processed and feature inferences were made. Morphological features were obtained for each grain of rice. \
When  looking  at  the  general  characteristics  of  Osmancik species, they have a wide, long, glassy and dull appearance.  
When looking at the general characteristics of the Cammeo species, they have wide and long, glassy and dull in appearance. \     
The data provided is sufficient to make an informed decision for each application.
"""