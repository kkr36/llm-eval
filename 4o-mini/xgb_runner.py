import os
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
import xgboost as xgb
import pandas as pd
from pathlib import Path
import importlib
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
data_folder = Path("data")

if __name__ == "__main__":

    auc_per_dataset = {}

    for f in tqdm(os.listdir(data_folder)):
        print(f)
        data = pd.read_csv(data_folder / f)
        # data = data.loc[:, ~data.columns.str.contains("Unnamed")]

        data[data.select_dtypes(include=['object']).columns] = data.select_dtypes(include=['object']).astype('category')

        # get the outcome variable from data encoding
        dataset_name = f.split(".csv")[0]
        data_encodings = importlib.import_module(f"data_encs.{dataset_name}")
        outcome = data_encodings.OUTCOMES[0]
        column_encodings = data_encodings.ColumnsEncoding
        columns_map: dict[str, object] = {
            col_mapper.value.name: col_mapper.value for col_mapper in column_encodings
        }

        # subset to include only columns in the data_encs file
        usable_features = set(columns_map[x]._name for x in columns_map)
        usable_features.add(outcome)
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

        y_train = train[outcome]
        x_train = train.drop(outcome, axis="columns")
        y_test = test[outcome]
        x_test = test.drop(outcome, axis="columns")

        pre_augmentation_model = XGBClassifier(enable_categorical=True, random_state=42)

        pre_augmentation_model.fit(x_train, y_train)

        probs = pre_augmentation_model.predict_proba(x_test)[:,1]

        xgb.plot_importance(pre_augmentation_model)
        plt.savefig(f"xgb_importance/{f.split('.')[0]}.png")

        # collect auc
        auc = roc_auc_score(y_test.to_numpy(), probs)
        if auc == 1:
            import pdb; pdb.set_trace()
        auc_per_dataset[dataset_name] = auc
    
    # convert aucs to df
    df = pd.DataFrame(list(auc_per_dataset.items()), columns=['dataset_name', 'auc'])
    df.to_csv('xgb.csv', index=False)