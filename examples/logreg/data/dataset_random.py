#!/usr/bin/env python3

import pandas as pd
import sklearn as sk

# Create random dataset.
x, y = sk.datasets.make_blobs(n_samples=100000, centers=2, cluster_std=1.0, random_state=42, n_features=25)

scaler = sk.preprocessing.StandardScaler()
x = scaler.fit_transform(x)
scaler_2 = sk.preprocessing.MinMaxScaler(feature_range=(0, 0.5))
x = scaler_2.fit_transform(x)

# Save the random dataset.
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle=True)

train = pd.DataFrame(x_train)
train['target'] = y_train.astype(float)
test = pd.DataFrame(x_test)
test['target'] = y_test.astype(float)

train.to_csv('random_data_train.csv', index=False, header=False)
test.to_csv('random_data_validation.csv', index=False, header=False)
