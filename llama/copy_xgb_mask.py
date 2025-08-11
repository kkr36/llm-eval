import shutil
import os
from pathlib import Path

confidence_scores_dir = Path("results")

if __name__ == "__main__":
    for f in os.listdir(confidence_scores_dir):
        assert(len(os.listdir(confidence_scores_dir / f)) == 1)
        timestamp = os.listdir(confidence_scores_dir / f)[0]
        openai_dir = Path(f"{confidence_scores_dir}/{f}/{timestamp}")
        masking_pickle = openai_dir / "all_results.pickle"
        shutil.copy(masking_pickle, Path("xgb_pickles_llama") / f"{f.split('_masking')[0]}.pickle")