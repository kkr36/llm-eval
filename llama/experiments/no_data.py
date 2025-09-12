import os
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
from folktexts.task import TaskMetadata
from openai import OpenAI
from tqdm import tqdm

from experiments.llama_util import generate_probs


def process_batch(file_response):
    num_weights = len(file_response)
    weights = []

    for response in tqdm(file_response):
        try:
            res = response.split("Guess:")[-1]
            weights.append(float(res))
        except:
            pdb.set_trace()
    assert len(weights) == num_weights
    return weights


class Experiment:
    def __init__(
        self,
        model,
        data,
        column_encodings,
        reentries,
        random_seed,
        question,
        config,
        experiment_name,
        outcome,
        context,
    ):
        self.model = model
        self.train_csv = data
        self.outcome = outcome
        ## NEW
        self.column_encodings = column_encodings
        self.reentries = reentries
        self.random_seed = random_seed
        self.question = question
        self.config = config
        self.context = context
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.experiment_name = experiment_name

    def get_weights(self, artifacts_dir):
        client = OpenAI(api_key=self.api_key)

        train_indices = [0]
        train = self.train_csv.iloc[train_indices].reset_index(drop=True)
        for col in train.columns:
            train[col] = "<val>"

        columns_map: dict[str, object] = {
            col_mapper.value.name: col_mapper.value for col_mapper in self.column_encodings
        }

        reentry_qa = self.reentries.reentry_qa.value
        reentry_numeric_qa = self.reentries.reentry_numeric_qa.value
        features = [feature for feature in columns_map.keys() if feature not in self.outcome]

        reentry_task = TaskMetadata(
            name="income prediction",
            description=self.context,
            features=features,
            target=self.outcome[0],
            cols_to_text=columns_map,
            sensitive_attribute=None,
            multiple_choice_qa=reentry_qa,
            direct_numeric_qa=reentry_numeric_qa,
        )

        s = reentry_task.get_row_description_blank(train.iloc[train_indices[0]])

        # pass each string through gpt; ask to reweight

        format_str = (
            "an integer from 1 to 5"
            if "score" in self.experiment_name
            else "a decimal from 0.0 to 1.0, without any leading 0; your first character should be a decimal place '.', followed by numeric characters"
        )

        context = f"Consider the following description of a dataset: {self.context}. One example entry from this dataset might look like the following: {s}. The goal of the classification task is to predict the following, for this dataset: {self.question} Please respond with {format_str} only."

        if "score" in self.experiment_name:
            question = f"On a scale from 1 to 5, where 5 means you are very confident in your knowledge of the domain and 1 indicates no confidence, how confident are you that you will predict a given datapoint's outcome ({self.outcome[0]}) correctly, for this dataset?"
        elif "auc" in self.experiment_name:
            question = f"If you were to predict the outcome ({self.outcome[0]}) for many rows from this dataset, provide a point estimate for what you think your resulting AUC compared to the true labels would be, for this dataset."
        else:
            assert "probs" in self.experiment_name
            question = f"Expressed as a decimal between 0 and 1, written without a leading zero (first character should be '.'), where higher values mean you are very confident in your knowledge of the domain and low values indicates less confidence, how confident are you that you will predict a given datapoint's outcome ({self.outcome[0]}) correctly, for this dataset?"

        prompt = f"""Provide your best guess, formatted as {format_str}, for the following question. \
        Give ONLY the guess, no other words or explanation. For example:
        \n\n<most likely guess, as short as possible; not a complete sentence, just the guess!>
        \n\nThe question is: {question}"""

        results = []

        if "openai" in self.model:
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ]

            response = client.chat.completions.create(
                model=self.model.split("/")[-1],
                messages=messages,
                max_tokens=3,
                logprobs=True,
                top_logprobs=10,
                temperature=0,
            )

            if "score" in self.experiment_name:
                assert len(response.choices[0].message.content) == 1
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                numericals, probs = [], []
                for logprob_obj in logprobs:
                    try:
                        candidate_score = int(logprob_obj.token)
                        candidate_prob = np.exp(logprob_obj.logprob)
                        numericals.append(candidate_score)
                        probs.append(candidate_prob)
                    except:
                        print(f"{logprob_obj.token} was not numerical; skipping")
                probs = [x / sum(probs) for x in probs]  # normalize by probs over numerical values
                score = np.dot(numericals, probs)
            else:
                # Skip decimal point; get next 2
                assert len(response.choices[0].logprobs.content) == 2
                logprobs = response.choices[0].logprobs.content[1].top_logprobs

                numericals, probs = [], []
                for logprob_obj in logprobs:
                    try:
                        candidate_score = float(f"0.{logprob_obj.token}")
                        candidate_prob = np.exp(logprob_obj.logprob)
                        numericals.append(candidate_score)
                        probs.append(candidate_prob)
                    except:
                        print(f"{logprob_obj.token} was not numerical; skipping")
                probs = [x / sum(probs) for x in probs]  # normalize by probs over numerical values
                score = np.dot(numericals, probs)
        else:
            assert "llama" in self.model
            full_prompt = (
                context + " " + prompt if "score" in self.experiment_name else context + " " + prompt + "answer: ."
            )
            candidate_probs = generate_probs(self.model, full_prompt)
            numericals, probs = [], []
            for token, candidate_prob in candidate_probs:
                try:
                    candidate_score = int(token) if "score" in self.experiment_name else float(f"0.{token}")
                    numericals.append(candidate_score)
                    probs.append(candidate_prob)
                except:
                    print(f"{token} was not numerical; skipping")
            probs = [x / sum(probs) for x in probs]  # normalize by probs over numerical values
            score = np.dot(numericals, probs)

        df = pd.DataFrame({"dataset_name": self.config.dataset, "rating": score}, index=[0])
        output_str = Path("results") / f"""{self.experiment_name.split("_")[-1]}_logprob.csv"""
        if not os.path.exists(output_str):
            df.to_csv(output_str)
        else:
            old_df = pd.read_csv(output_str)
            new_df = pd.concat([old_df, df], ignore_index=True)
            new_df = new_df.loc[:, ~new_df.columns.str.contains("^Unnamed")]
            new_df.to_csv(output_str)
