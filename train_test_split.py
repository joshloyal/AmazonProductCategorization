import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


target_values = ['Clothing', 'Shoes', 'Watches', 'Jewelry']
df = pd.read_csv('amazon_products.csv')
column_names = df.columns
df = df[df['product_category'].isin(target_values)]

y = df.pop('product_category').values
X = df.values

# intial train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)


# split train into train-dev datasets
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=123)

# save train/dev/test splits to a file
train_df = pd.DataFrame(np.hstack((X_train, y_train[:, np.newaxis])), columns=column_names)
print(train_df['product_category'].value_counts())
dev_df = pd.DataFrame(np.hstack((X_dev, y_dev[:, np.newaxis])), columns=column_names)
print(dev_df['product_category'].value_counts())
test_df = pd.DataFrame(np.hstack((X_test, y_test[:, np.newaxis])), columns=column_names)
print(test_df['product_category'].value_counts())

train_df.to_csv('amazon_products_train.csv', index=False)
dev_df.to_csv('amazon_products_dev.csv', index=False)
test_df.to_csv('amazon_products_test.csv', index=False)
