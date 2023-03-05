import pandas as pd
from sklearn.model_selection import KFold


def split_data(data: list, n_split: int):
    df = pd.DataFrame(data)
    kfold = KFold(n_splits=n_split, random_state=42, shuffle=True)
    for i, (train_index, val_index) in enumerate(kfold.split(df)):
        df.loc[val_index, "fold"] = i
    data_dict = df.to_dict("records")
    return data_dict
