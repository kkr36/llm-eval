from string import Template


THE_QUESTION = "Was the patient readmitted to the hospital within the next 30 days?"

TASK_TEMPLATE = Template(
    """Provide your best guess and the probability that it is correct (0.0 to 1.0) for
        the following question. Give ONLY the guess and probability, no other words or
        explanation. For example:\n\nGuess: <most likely guess, as short as possible; not
        a complete sentence, just the guess!>\n Probability: <the probability between 0.0
        and 1.0 that your guess is correct, without any extra commentary whatsoever; just
        the probability!>\n\nThe question is: $the_question"""
)

TASK_DESCRIPTION = TASK_TEMPLATE.substitute(the_question=THE_QUESTION)

CONTEXT = """The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. Information was extracted from the database for encounters that satisfied the following criteria.
(1)	It is an inpatient encounter (a hospital admission).
(2)	It is a diabetic encounter, that is, one during which any kind of diabetes was entered into the system as a diagnosis.
(3)	The length of stay was at least 1 day and at most 14 days.
(4)	Laboratory tests were performed during the encounter.
(5)	Medications were administered during the encounter. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.""".replace("\n", "")