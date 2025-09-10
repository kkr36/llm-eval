import os
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
from folktexts.task import TaskMetadata
from openai import OpenAI
from tqdm import tqdm


def process_batch(file_response):
    num_weights = len(file_response)
    weights = []

    for response in tqdm(file_response):
        try:
            _, res = response.split("Probability:")
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
        task_prompt,
        config,
        experiment_name,
        outcome,
        context,
        prompt,
        # process_batch_fn
    ):
        self.model = model
        self.train_csv = data
        self.outcome = outcome
        ## NEW
        self.column_encodings = column_encodings
        self.reentries = reentries
        self.random_seed = random_seed
        self.task_prompt = task_prompt
        self.config = config

        self.prompt = prompt
        # self.process_batch = process_batch_fn
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.context = context
        self.experiment_name = experiment_name

    def get_weights(self, artifacts_dir):
        client = OpenAI(api_key=self.api_key)

        # turn the df to text and save

        np.random.seed(self.random_seed)
        train_indices = np.random.choice(len(self.train_csv), size=min(1000, len(self.train_csv)), replace=False)

        train = self.train_csv.iloc[train_indices].reset_index(drop=True)

        columns_map: dict[str, object] = {
            col_mapper.value.name: col_mapper.value for col_mapper in self.column_encodings
        }

        reentry_qa = self.reentries.reentry_qa.value
        reentry_numeric_qa = self.reentries.reentry_numeric_qa.value
        features = [feature for feature in columns_map.keys() if feature not in self.outcome]

        reentry_task = TaskMetadata(
            name="income prediction",
            description=self.task_prompt,
            features=features,
            target=self.outcome[0],
            cols_to_text=columns_map,
            sensitive_attribute=None,
            multiple_choice_qa=reentry_qa,
            direct_numeric_qa=reentry_numeric_qa,
        )

        strs = [reentry_task.get_row_description(train.iloc[i]) for i in range(len(train))]

        # pass each string through gpt; ask to reweight

        context = self.context

        prompt = self.prompt

        results = []

        for s in tqdm(strs):
            full_prompt = f"{s} {prompt}"
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": full_prompt},
            ]

            pdb.set_trace()

            response = client.chat.completions.create(model=self.model.split("/")[-1], messages=messages)
            results.append(response.choices[0].message.content)

        uncertainty_scores = process_batch(results)
        df = pd.DataFrame({"uncertainty_score": uncertainty_scores})
        output_str = artifacts_dir / f"{self.experiment_name}"
        output_path = Path(output_str)
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_str / "uncertainty_scores.csv", index=False)
