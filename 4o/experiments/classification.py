from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import Benchmark
from folktexts.dataset import Dataset
from folktexts.task import TaskMetadata


# TODO Perhaps this will evolve into an abstract class?


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
):
    """_summary_

    Args:
        model (_type_): _description_
        artifacts_dir (_type_): _description_
        data (_type_): _description_
    """

    num_data = len(data)

    if num_data > 1000:
        subsampling = (1000 / 0.95) / num_data
    else:
        subsampling = 1.

    columns_map: dict[str, object] = {
        col_mapper.value.name: col_mapper.value for col_mapper in column_encodings
    }

    reentry_qa = reentries.reentry_qa.value
    reentry_numeric_qa = reentries.reentry_numeric_qa.value
    features = [feature for feature in columns_map.keys() if feature not in outcomes]

    reentry_task = TaskMetadata(
        name="income prediction",
        description=task_prompt,
        features=features,
        target=outcomes[0],
        cols_to_text=columns_map,
        sensitive_attribute=None,
        multiple_choice_qa=reentry_qa,
        direct_numeric_qa=reentry_numeric_qa,
    )


    # TODO We need to handle the subsampling way better
    reentry_dataset = Dataset(
        data=data,
        task=reentry_task,
        # more burned parameters
        test_size=0.95,
        val_size=0,
        subsampling=subsampling,  # NOTE: Optional, for faster but noisier results!
        seed=random_seed,
    )

    all_tasks = {"reentry": [reentry_task, reentry_dataset]}

    for taskname in all_tasks:
        task, dataset = all_tasks[taskname]
        llm_clf = WebAPILLMClassifier(model_name=model, task=task, custom_prompt_prefix=task_prompt)
        llm_clf.set_inference_kwargs(batch_size=500)
        bench = Benchmark(llm_clf=llm_clf, dataset=dataset)
        # think how to abstract away the dir path to avoid bugs
        bench.run(results_root_dir=(artifacts_dir / taskname))
