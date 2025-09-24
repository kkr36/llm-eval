import argparse
import os
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="takes in run name along with config parameters in grid form")
    parser.add_argument("run_name", type=str, help="Name of the run (required)")

    args = parser.parse_args()

    if not os.path.exists(f"configs/{args.run_name}"):
        raise ValueError("run dir does not exist!")

    print(f"Run name provided: {args.run_name}")

    for json_path in os.listdir(f"configs/{args.run_name}"):
        dataset_name = json_path.split("_")[0]
        if dataset_name in [x.split("_")[0] for x in os.listdir("results")]:
            continue
        print(f"RUNNING CONFIG {json_path}")
        subprocess.run(["python", "main.py", f"configs/{args.run_name}/{json_path}"])
