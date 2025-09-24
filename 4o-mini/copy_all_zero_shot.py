import os
import shutil
from pathlib import Path


if __name__ == "__main__":
    masking_dir = Path("past_results/gpt_masking")
    zero_shot_dir = Path("past_results/gpt_zero_shot")
    output_dir = Path("all_zero_shot")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # copy over all the zero shot
    for f in os.listdir(zero_shot_dir):
        assert len(os.listdir(zero_shot_dir / f)) == 1
        timestamp = os.listdir(zero_shot_dir / f)[0]
        openai_dir = Path(f"{zero_shot_dir}/{f}/{timestamp}/reentry/openai")
        assert len(os.listdir(openai_dir)) == 1
        bench_dir = os.listdir(openai_dir)[0]
        results_dir = openai_dir / bench_dir
        shutil.copytree(results_dir, output_dir / bench_dir)

    # copy over all the masking
    for f in os.listdir(masking_dir):
        assert len(os.listdir(masking_dir / f)) == 1
        timestamp = os.listdir(masking_dir / f)[0]
        timestamp_dir = Path(f"{masking_dir}/{f}/{timestamp}")
        for exp_f in os.listdir(timestamp_dir):
            if ".pickle" in exp_f or ".json" in exp_f:
                continue
            openai_dir = timestamp_dir / exp_f / "openai"
            assert len(os.listdir(openai_dir)) == 1
            bench_dir = os.listdir(openai_dir)[0]
            results_dir = openai_dir / bench_dir
            shutil.copytree(results_dir, output_dir / bench_dir)
