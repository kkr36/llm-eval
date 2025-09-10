import importlib
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from config import Config
def run_experiment(config: Config):
    experiment_name = config.experiment
    exp_import_str = (
        "no_data"
        if "auc" in experiment_name or "score" in experiment_name or "probs" in experiment_name
        else experiment_name
    )
    experiment = importlib.import_module(f"""experiments.{exp_import_str}""")
    model = config.model
    dataset = config.dataset
    data = pd.read_csv(f"../data/{dataset}.csv")

    
    # Is there a better way to use the API keys?
    with open("secrets.json", "r") as f:
        keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = keys["open_ai_key"]
    os.environ["HUGGINGFACE_TOKEN"] = keys["huggingface_token"]

    timestamp = pd.Timestamp.now().strftime("%m%d%H%M")
    artifacts_dir = Path("results") / f"{dataset}_{experiment_name}" / timestamp
    if exp_import_str != "no_data":
        artifacts_dir.mkdir(exist_ok=True, parents=True)
    task_prompts = importlib.import_module(f"task_prompts.{config.task_prompt}")
    data_encodings = importlib.import_module(f"data_encs.{config.column_encodings}")
    column_encodings = data_encodings.ColumnsEncoding
    reentries = data_encodings.Reentry
    outcomes = data_encodings.OUTCOMES

    if "auc" in experiment_name or "probs" in experiment_name or "score" in experiment_name:
        task_prompts = importlib.import_module(f"task_prompts.{config.task_prompt}")
        question = task_prompts.THE_QUESTION
        context = task_prompts.CONTEXT
        experiment = importlib.import_module(f"experiments.no_data").Experiment(
            model=model,
            # artifacts_dir=artifacts_dir,
            data=data,
            column_encodings=column_encodings,
            reentries=reentries,
            # outcomes=outcomes,
            random_seed=int(config.random_seed),
            question=question,
            config=config,
            # model=config.model,
            # train_csv_path=data_encodings.train_csv_path,
            # column_ordering=data_encodings.column_ordering,
            # discrete_columns=data_encodings.discrete_columns,
            experiment_name=experiment_name,
            context=context,
            outcome=outcomes,
            # serialization_fn=data_encodings.serialization_fn,
            # additional_info_fn=data_encodings.additional_info_fn,
            # prompt=task_prompt,
            # process_batch_fn=data_encodings.process_batch_fn
        )
        experiment.get_weights(artifacts_dir)
    elif "masking" in experiment_name:
        execute_experiment = getattr(experiment, "execute_experiment")
        execute_experiment(
            model=model,
            artifacts_dir=artifacts_dir,
            data=data,
            column_encodings=column_encodings,
            reentries=reentries,
            outcomes=outcomes,
            random_seed=int(config.random_seed),
            task_prompt=task_prompts.TASK_DESCRIPTION,
            config=config,
            discretize_cols=data_encodings.discretize_cols,
        )
    elif config.use_folktexts:
        execute_experiment = getattr(experiment, "execute_experiment")
        execute_experiment(
            model=model,
            artifacts_dir=artifacts_dir,
            data=data,
            column_encodings=column_encodings,
            reentries=reentries,
            outcomes=outcomes,
            random_seed=int(config.random_seed),
            task_prompt=task_prompts.TASK_DESCRIPTION,
            config=config,
        )
    else:
        assert "confidence" in experiment_name
        task_prompts = importlib.import_module(f"task_prompts.{config.task_prompt}")
        task_prompt = task_prompts.TASK_DESCRIPTION
        task_context = task_prompts.CONTEXT
        experiment = importlib.import_module(f"experiments.{experiment_name}").Experiment(
            model=model,
            # artifacts_dir=artifacts_dir,
            data=data,
            column_encodings=column_encodings,
            reentries=reentries,
            # outcomes=outcomes,
            random_seed=int(config.random_seed),
            task_prompt=task_prompts.TASK_DESCRIPTION,
            config=config,
            # model=config.model,
            # train_csv_path=data_encodings.train_csv_path,
            # column_ordering=data_encodings.column_ordering,
            # discrete_columns=data_encodings.discrete_columns,
            experiment_name=experiment_name,
            outcome=outcomes,
            context=task_context,
            # serialization_fn=data_encodings.serialization_fn,
            # additional_info_fn=data_encodings.additional_info_fn,
            prompt=task_prompt,
            # process_batch_fn=data_encodings.process_batch_fn
        )
        experiment.get_weights(artifacts_dir)

    if exp_import_str != "no_data":
        with open(artifacts_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Results saved in {artifacts_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run experiments with JSON configuration.")
    parser.add_argument("config", help="Path to the JSON configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = json.load(f)

    config = Config(**config_dict)

    run_experiment(config)
