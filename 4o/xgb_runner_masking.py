import os
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
import xgboost as xgb
import pandas as pd
from pathlib import Path
import importlib
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import pickle
data_folder = Path("data")

if __name__ == "__main__":

    auc_per_dataset = {}

    for f in tqdm(os.listdir(data_folder)):
        print(f)
        data = pd.read_csv(data_folder / f)

        data[data.select_dtypes(include=['object']).columns] = data.select_dtypes(include=['object']).astype('category')

        # get the outcome variable from data encoding
        dataset_name = f.split(".csv")[0]
        data_encodings = importlib.import_module(f"data_encs.{dataset_name}")
        outcome = data_encodings.OUTCOMES[0]
        column_encodings = data_encodings.ColumnsEncoding
        discretize_cols = data_encodings.discretize_cols
        columns_map: dict[str, object] = {
            col_mapper.value: col_mapper.value for col_mapper in column_encodings
        }

        # subset to include only columns in the data_encs file
        usable_features = set(columns_map[x]._name for x in columns_map)
        usable_features.remove(outcome)
        usable_features = list(usable_features)
        data = data[usable_features]

        # Characters to replace
        banned_chars = [']', '[', ',', '<']

        # Replace each banned character with an underscore (or any other character)
        def clean_column(col):
            for ch in banned_chars:
                col = col.replace(ch, '_')  # Replace with underscore
            return col
        
        # Apply to all column names
        data.columns = [clean_column(col) for col in data.columns]

        # get data and train xgb

        # Set size limit
        subset_size = min(1000, len(data) // 2)

        # Randomly sample training set
        train_df = data.sample(n=subset_size, random_state=42)

        # Drop training rows to get validation set
        val_df = data.drop(train_df.index).sample(n=subset_size, random_state=43)

        # Optional: reset indices if desired
        train = train_df.reset_index(drop=True)
        test = val_df.reset_index(drop=True)

        print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

        # for each dataset, get each sampled outcome variable
        mask_dir = Path(f"results")
        exp_name = f"{f.split('.')[0]}_masking"
        assert(len(os.listdir(mask_dir / exp_name)) == 1)
        timestamp = os.listdir(mask_dir / exp_name)[0]
        openai_dir = Path(f"{mask_dir}/{exp_name}/{timestamp}")
        masking_pickle = openai_dir / "all_results.pickle"
        with open(masking_pickle, 'rb') as handle:
            masking_results = pickle.load(handle)
        
        used_cols = [''.join(col.split("_binary")[:-1]) for col in masking_results]

        # iterate through each sampled variable, make the binary version, and run xgb on the binary version using all other variables
        for real_col in used_cols:
            col = real_col
            for banned_char in banned_chars:
                col = col.replace(banned_char, '_')

            filtered_data = data[data[col].notnull()]
            filtered_train, filtered_test = train[train[col].notnull()], test[test[col].notnull()] # TODO add unique identifier

            options_set = list(set(data[col].tolist()))
            final_col_name_full = f"{real_col}_binary"
            # clean up colnames from the pkl, to fit with xgb naming requirement
            final_col_name = final_col_name_full
            for banned_char in banned_chars:
                final_col_name = final_col_name.replace(banned_char, '_')

            if (
                len(options_set) == 2
            ):  # already binary; use the 2 options to make a yes/no question
                positive, negative = max(options_set), min(options_set)
                filtered_train[final_col_name] = (
                    filtered_train[col] == positive
                ).astype(int)  # positive -> 1; negative -> 0
                filtered_test[final_col_name] = (
                    filtered_test[col] == positive
                ).astype(int)  # positive -> 1; negative -> 0

            elif col in discretize_cols:  # numerical value; use the median
                try:
                    if ("datetime" not in col) and ("_Time" not in col):
                        median = np.median(filtered_data[col])
                    else:
                        filtered_data[col] = pd.to_datetime(filtered_data[col])
                        filtered_train[col] = pd.to_datetime(filtered_train[col])
                        filtered_test[col] = pd.to_datetime(filtered_test[col])
                        median = filtered_data[col].median()
                except:
                    import pdb; pdb.set_trace()
                try:
                    filtered_train[final_col_name] = (filtered_train[col] > median).astype(
                        int
                    )  # if median above, say 1; otherwise 0
                    filtered_test[final_col_name] = (filtered_test[col] > median).astype(
                        int
                    )  # if median above, say 1; otherwise 0
                except:
                    import pdb; pdb.set_trace()

            else:  # categorical variable; take the mode and binarize this
                mode = filtered_data[col].mode().iloc[0]
                filtered_train[final_col_name] = (filtered_train[col] == mode).astype(
                    int
                )  # if equal to mode say 1; otherwise 0
                filtered_test[final_col_name] = (filtered_test[col] == mode).astype(
                    int
                )  # if equal to mode say 1; otherwise 0

            train_new = filtered_train.drop(col, axis="columns") # get rid of original col, which was used to make the binary outcome
            test_new = filtered_test.drop(col, axis="columns") # get rid of original col, which was used to make the binary outcome

            y_train = train_new[final_col_name]
            x_train = train_new.drop(final_col_name, axis="columns")
            y_test = test_new[final_col_name]
            x_test = test_new.drop(final_col_name, axis="columns")

            pre_augmentation_model = XGBClassifier(enable_categorical=True, random_state=42)

            pre_augmentation_model.fit(x_train, y_train)

            probs = pre_augmentation_model.predict_proba(x_test)[:,1]

            # collect auc
            auc = roc_auc_score(y_test.to_numpy(), probs)
            # if auc == 1:
            #     import pdb; pdb.set_trace()

            # add it back to the pickle and resave?
            masking_results[final_col_name_full]["xgb_auc"] = auc
        # import pdb; pdb.set_trace()
        with open(Path("xgb_pickles_4o") / f"{dataset_name}.pickle", "wb") as g:
            pickle.dump(masking_results, g)
