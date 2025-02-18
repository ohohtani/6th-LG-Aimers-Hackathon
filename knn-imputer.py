import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# bring data
train_DF = pd.read_csv("data/train.csv")
test_DF = pd.read_csv("data/test.csv")

train_DF = train_DF.drop(["ID", "배란 유도 유형"], axis=1)

# labeling
label_encoder = LabelEncoder()

columns = train_DF.columns
for col in columns:
    if train_DF[col].dtype == object:
        train_DF[col] = label_encoder.fit_transform(train_DF[col])


# knn-imputing
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(train_DF), columns=train_DF.columns)


df_imputed.to_csv('data/knn_preprocessed.csv', index=False)