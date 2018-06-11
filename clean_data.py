import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('datasets/kdd_original10.data', header=None)
dataset = dataset.rename(columns = {41:'label'})
dataset_labels = dataset.label
dataset = dataset.drop('label', axis=1)

for column in dataset:
    if (dataset[column].dtype == 'object'):
        dataset[column] = dataset[column].astype('category')
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column]).astype(float)
    else:
        dataset[column] = dataset[column].astype(float)

dataset.insert(0, 42, dataset_labels)
dataset.to_csv('datasets/kdd99.data', index=False, header=False)