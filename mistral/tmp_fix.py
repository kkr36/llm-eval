import shutil
import os

if __name__ == "__main__":
    res_dir = "results"
    for subdir in os.listdir(res_dir):
        if "classification" in subdir:
            shutil.move(f"{res_dir}/{subdir}", "llama_zero_shot")
        else:
            assert("confidence" in subdir)
            shutil.move(f"{res_dir}/{subdir}", "llama_confidence")
    