# %% import statements
import pandas as pd
import pickle
from sklearn import preprocessing

# %% load dataset
df = pd.read_excel('bankloan_5000.xlsx', sheet_name=0)
print(df)

# get training and test dfs
train_df = df[:4000]
test_df = df[4000:]

# %% load models
RF_model = pickle.load(open('RF_model.h5', 'rb'))
LR_model = pickle.load(open('LR_model.h5', 'rb'))

# %% preprocess data as done in training
# save copy of test dataframe before preprocessing
original_test_df = test_df.copy()

test_df['creddebt'] = test_df['creddebt'].apply(lambda x: round(x / 1000, 2))

# drop default column
test_df = test_df.drop(columns=['default'])
print(test_df)

# %% predict
RF_y_pred = RF_model.predict(test_df)
print(f"Random Forest Predictions:\n{RF_y_pred}")

# further preprocessing(Scaling of data) for logistic regression
cols = test_df.columns
scaled_data = preprocessing.StandardScaler().fit_transform(test_df)
scaled_test_df = pd.DataFrame(scaled_data, columns=cols)

LR_y_pred = LR_model.predict(scaled_test_df)
print(f"Random Forest Predictions:\n{LR_y_pred}")

# %% save predictions to dataframe
original_test_df.drop(columns=['default'], inplace=True)

original_test_df['random_forest_predictions'] = RF_y_pred
original_test_df['logistic_regression_predictions'] = LR_y_pred

# save to csv
original_test_df.to_csv('Prospective Customers Predictions.csv', index=False)
