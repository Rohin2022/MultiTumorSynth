import pandas as pd


all_data = pd.read_csv("mask_metrics_v7.csv")
swiss = pd.read_csv("/projects/bodymaps/Data/metadata_swiss.csv")
turkish = pd.read_csv("/projects/bodymaps/Data/turkish_dataset_meta_latest.csv")

indices = all_data["bdmap_id"].isin(pd.concat([swiss["BDMAP ID"], turkish["BDMAP ID"]]))
test = all_data[indices]
train = all_data[~indices]


test.to_csv("AAProTest.csv",index=False)
train.to_csv("AAProTrain.csv",index=False)