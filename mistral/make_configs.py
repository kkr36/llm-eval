import json
import os
import argparse

# llms = ['openai/gpt-4o-mini']
llms = ['mistralai/Mistral-7B-Instruct-v0.1']
experiments = ['classification', 'confidence', 'masking'][2:3]
# experiments = ['probs', 'auc', 'score']
datasets = [
    'acsincome',
    'acsmobility',
    'acspubcov',
    'acstraveltime',
    'acsunemployment',
    'airline',
    'bank',
    'brfssdiabetes',
    'brfsshbp',
    'brfsshighcholesterol',
    'car',
    'diabetes',
    'glioma',
    'houses',
    'IndianDiabetes',
    'ipums',
    'mushroom',
    'nursery',
    'rice',
    'sepsis',
    'support2',
    'taxibog',
    'taximex',
    'taxiuio',
    'telescope',
    'ucibreastcancer',
    'ucidiabetes',
    'uciheart',
    'ucispambase',
    'ucistatloggerman',
    'usaccidents'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="takes in run name along with config parameters in grid form")
    parser.add_argument(
        "run_name",
        type=str,
        help="Name of the run (required)"
    )

    args = parser.parse_args()

    if os.path.exists(f"configs/{args.run_name}"):
        raise ValueError("run dir already exists!")
    os.makedirs(f"configs/{args.run_name}")

    print(f"Run name provided: {args.run_name}")

    existing_runs = os.listdir('results')
    existing_runs = [x.split("_")[0] for x in existing_runs]

    for dataset in datasets:
        if dataset not in existing_runs:
            for llm in llms:
                for experiment in experiments:
                    obj = {}
                    obj["experiment"] = experiment
                    obj["column_encodings"] = dataset
                    obj["dataset"] = dataset
                    obj["model"] = llm
                    obj["use_folktexts"] = True if "confidence" not in experiment else False
                    obj["random_seed"] = 42
                    obj["task_prompt"] = f"{'confidence' if 'classification' not in experiment else 'classification'}.{dataset}"

                    with open(f"configs/{args.run_name}/{dataset}_{experiment}_{llm.split('/')[-1]}.json", 'w') as f:
                        json.dump(obj, f, indent=4)