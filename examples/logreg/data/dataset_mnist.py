#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sklearn as sk

# MNIST Data Download.
mnist = sk.datasets.fetch_openml('mnist_784')
x = pd.DataFrame(mnist.data)
y = pd.DataFrame(mnist.target).astype('int')

# Filter by 2 given digits.
x_38 = x[np.any([y == 1,y == 8], axis = 0)].reset_index(drop = True)
y_38 = y[np.any([y == 1,y == 8], axis = 0)].reset_index(drop = True)

# Map digits to 1 or 0.
y_38[y_38 == 1] = 0
y_38[y_38 == 8] = 1

# Process the data.
scaler = sk.preprocessing.StandardScaler()
x_38 = scaler.fit_transform(x_38)
scaler_2 = sk.preprocessing.MinMaxScaler(feature_range=(0, 0.2))
x_38 = scaler_2.fit_transform(x_38)

# Scale down images.
x_38 = pd.DataFrame(x_38)
delete_cols = False
for i in range(0, 784, 28):
    if delete_cols:
        x_38 = x_38.drop([i for i in range(i, (i+28))], axis='columns')
    else:
        x_38 = x_38.drop([i for i in range(i+1, (i+28), 2)], axis='columns')
    delete_cols = not delete_cols

x_38 = x_38.to_numpy()

# Partition the original data.
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x_38, y_38, test_size = 0.2, random_state = 42, shuffle=True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)

# Associate label and samples dataframes.
train = pd.DataFrame(x_train)
train['target'] = y_train.astype(float)
test = pd.DataFrame(x_test)
test['target'] = y_test.astype(float)

# Output CSV using pandas' to_csv()
train.to_csv('mnist_data_train.csv', index=False, header=False)
test.to_csv('mnist_data_validation.csv', index=False, header=False)
