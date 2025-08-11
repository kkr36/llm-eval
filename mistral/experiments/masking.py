import pickle

import numpy as np
import pandas as pd
import random

from folktexts.benchmark import Benchmark
from folktexts.classifier import WebAPILLMClassifier
from folktexts.classifier import TransformersLLMClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
login(os.getenv("HUGGINGFACE_TOKEN"))
from folktexts.col_to_text import ColumnToText
from folktexts.dataset import Dataset
from folktexts.qa_interface import Choice, DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata


def execute_experiment(
    model,
    artifacts_dir,
    data,
    column_encodings,
    reentries,
    outcomes,
    random_seed,
    task_prompt,
    config,
    discretize_cols
):
    """_summary_

    Args:
        model (_type_): _description_
        artifacts_dir (_type_): _description_
        data (_type_): _description_
    """

    # iterate through randomly selected columns (or possible all columns? tbd)
    # for each column calculate auc; take an average

    ban_cols = [outcomes[0]] # whole point is to not use outcome col
    # if "subsampling" in config.additional_params:
    #     subsampling = (float(config.additional_params["subsampling"]) / 0.95) / len(
    #         data
    #     )
    num_data = len(data)

    if num_data > 1000:
        subsampling = (1000 / 0.95) / num_data
    else:
        subsampling = 1.

    for col in data.columns:
        # apply null threshold
        # if data[col].isnull().mean() >= float(
        #     config.additional_params["null_threshold"]
        # ):
        if data[col].isnull().mean() >= 0.7:
            ban_cols.append(col)
            continue
        # can't be monolithic
        if len(set(data[col].tolist())) < 2:
            ban_cols.append(col)
            continue
        # can't be imbalanced to the point where subsampling induces nan aucs
        try:
            if data[col].value_counts(normalize=True).iloc[0] >= 0.99:
                print(col)
                ban_cols.append(col)
        except:
            # it's probably all nan; just ban it
            ban_cols.append(col)
            # import pdb; pdb.set_trace()
        # if categorical, mode has to be at least somewhat common
        if col not in discretize_cols and data[col].value_counts(normalize=True).iloc[0] <= 0.1:
            ban_cols.append(col)

    all_columns_map: dict[str, object] = {
        col_mapper.value.name: col_mapper.value for col_mapper in column_encodings
    }

    all_tasks = {}

    # randomly sample columns, in case there's too many covariates
    random.seed(config.random_seed)

    # TODO: figure out how to get all usable columns (all columns - ban_cols), then sample from this
    usable_columns = set(all_columns_map.keys())
    usable_columns = usable_columns.difference(set(ban_cols))
    usable_columns_map = {k: all_columns_map[k] for k in usable_columns}
    del all_columns_map[outcomes[0]]

    # sample_size = min(10, len(usable_columns_map))
    # sampled_cols = random.sample(sorted(list(usable_columns_map.items())), sample_size)

    import pickle
    with open(f"xgb_pickles/{config.dataset}.pickle", 'rb') as handle:
        gpt_res = pickle.load(handle)
    sampled_cols = [(''.join(key.split("_binary")[:-1]), usable_columns_map[''.join(key.split("_binary")[:-1])]) for key in gpt_res]
    # import pdb; pdb.set_trace()

    for col_name, col in sampled_cols:
        if col.name in ban_cols:
            continue  # skip the ones that can't be discretized

        tmp_map = all_columns_map.copy()
        filtered_data = data[data[col.name].notnull()]

        options_set = list(set(data[col.name].tolist()))
        final_col_name = f"{col.name}_binary"

        if (
            len(options_set) == 2
        ):  # already binary; use the 2 options to make a yes/no question
            positive, negative = max(options_set), min(options_set)
            filtered_data[final_col_name] = (
                filtered_data[col.name] == positive
            ).astype(int)  # positive -> 1; negative -> 0

            newCol = ColumnToText(
                final_col_name,
                short_description=col.short_description,
                value_map={0: col.value_map(negative), 1: col.value_map(positive)},
            )
            tmp_map[final_col_name] = newCol

            numeric_q = DirectNumericQA(
                column=final_col_name,
                text=(f"What is this person's {col.short_description}?"),
            )

            mc_q = MultipleChoiceQA(
                column=final_col_name,
                text=f"What is the value of {col.short_description}?",
                choices=(
                    Choice(col.value_map(positive), 1),
                    Choice(col.value_map(negative), 0),
                ),
            )

        elif col.name in discretize_cols:  # numerical value; use the median
            try:
                if ("datetime" not in col.name) and ("_Time" not in col.name):
                    median = np.median(filtered_data[col.name])
                else:
                    filtered_data[col.name] = pd.to_datetime(filtered_data[col.name])
                    median = filtered_data[col.name].median()
            except:
                import pdb; pdb.set_trace()
            filtered_data[final_col_name] = (filtered_data[col.name] > median).astype(
                int
            )  # if median above, say 1; otherwise 0

            newCol = ColumnToText(
                final_col_name,
                short_description=col.short_description,
                value_map={
                    0: f"less than or equal to {median}",
                    1: f"greater than {median}",
                },
            )
            tmp_map[final_col_name] = newCol

            numeric_q = DirectNumericQA(
                column=final_col_name,
                text=(f"What is this person's {col.short_description}?"),
            )

            med_val = median if isinstance(median, pd.Timestamp) else col.value_map(median)

            mc_q = MultipleChoiceQA(
                column=final_col_name,
                text=f"Is the value of {col.short_description} above or below/equal to {med_val}?",
                choices=(
                    Choice(f"greater than {med_val}", 1),
                    Choice(f"less than or equal to {med_val}", 0),
                ),
            )
        else:  # categorical variable; take the mode and binarize this
            mode = filtered_data[col.name].mode().iloc[0]
            filtered_data[final_col_name] = (filtered_data[col.name] == mode).astype(
                int
            )  # if equal to mode say 1; otherwise 0

            newCol = ColumnToText(
                final_col_name,
                short_description=col.short_description,
                value_map={0: f"not equal to {mode}", 1: f"equal to {mode}"},
            )
            tmp_map[final_col_name] = newCol

            numeric_q = DirectNumericQA(
                column=final_col_name,
                text=(f"What is this person's {col.short_description}?"),
            )

            mc_q = MultipleChoiceQA(
                column=final_col_name,
                text=f"Is the value of {col.short_description} equal to {col.value_map(mode)}?",
                choices=(
                    Choice(f"Yes, equal to {col.value_map(mode)}", 1),
                    Choice(f"No, not equal to {col.value_map(mode)}", 0),
                ),
            )

        all_outcomes = list(set([final_col_name, col.name]))

        task = TaskMetadata(
            name=f"{final_col_name} prediction",
            description=task_prompt,
            features=[x for x in all_columns_map.keys() if x not in all_outcomes],
            target=final_col_name,
            cols_to_text=tmp_map,
            sensitive_attribute=None,
            multiple_choice_qa=mc_q,
            direct_numeric_qa=numeric_q,
        )

        task.use_numeric_qa = (
            False  # TODO confirm this means we don't use the direct numeric question
        )

        dataset = Dataset(
            data=filtered_data,
            task=task,
            test_size=0.95,
            val_size=0,
            subsampling=subsampling,  # NOTE: Optional, for faster but noisier results!
            seed=random_seed,
        )

        all_tasks[final_col_name] = [task, dataset]

    all_results = {}

    if "mistral" in model:
        tokenizer = AutoTokenizer.from_pretrained(model, use_safetensors=True)
        llm = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True).to('cuda')
        tokenizer.pad_token_id = llm.config.eos_token_id

    for taskname in all_tasks:
        task, dataset = all_tasks[taskname]
        if "openai" in model:
            llm_clf = WebAPILLMClassifier(model_name=model, task=task, custom_prompt_prefix=task_prompt)
        else:
            assert("mistral" in model)
            llm_clf = TransformersLLMClassifier(model=llm, tokenizer=tokenizer, task=task, custom_prompt_prefix=task_prompt)
        llm_clf.set_inference_kwargs(batch_size=500 if "openai" in model else 8)

        # llm_clf = WebAPILLMClassifier(model_name=model, task=task, custom_prompt_prefix=task_prompt)
        # llm_clf.set_inference_kwargs(
        #     # batch_size=int(config.additional_params["batch_size"])
        #     batch_size=500
        # )
        bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

        RESULTS_DIR = artifacts_dir / taskname

        all_results[taskname] = bench.run(
            results_root_dir=RESULTS_DIR
        ) 

    avg_auc = np.mean(
        [
            all_results[key]["roc_auc"]
            for key in all_results
            if all_results[key]["roc_auc"] is not np.nan
        ]
    )

    with open(artifacts_dir / "all_results.pickle", "wb") as handle:
        pickle.dump(all_results, handle)

    # New row to add
    new_row = {
        "dataset_name": {config.dataset},
        "avg_auc": avg_auc,
    }

    # Check if file exists
    file_path = artifacts_dir / taskname / "dataset_results.csv"

    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["dataset_name", "avg_auc"])

    # Append new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save back to CSV
    df.to_csv(file_path, index=False)
