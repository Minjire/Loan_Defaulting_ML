# %% import statements
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from confusion_matrix_plot import plot_confusion_matrix
import sys
import pickle

# %% load dataset
df = pd.read_excel('bankloan_5000.xlsx', sheet_name=0)
print(df)

# get training and test dfs
train_df = df[:4000]
test_df = df[4000:]

# %% Data preprocessing
# convert credit card debt in thousands to match other monetary columns
train_df['creddebt'] = train_df['creddebt'].apply(lambda x: round(x / 1000, 2))
print(train_df.head())

# %%
# split data to features and target variable
X = train_df.drop(columns=['default'])
cols = X.columns
print(X.head())
y = train_df.default
print(y.head())

# %% split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# %% Modeling
RF = RandomForestClassifier(n_estimators=1000)
RF.fit(X_train, y_train)

# predict
y_pred = RF.predict(X_test)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

# %%
plot_confusion_matrix(matrix, classes=['default=0', 'default=1'], normalize=True,
                      title='Random Forest Confusion matrix')
plt.savefig("Random Forest Confusion matrix.png", bbox_inches='tight')
plt.show()
print(f"Confusion Matrix:\n{matrix}")
print(f"Classification Report:\n{report}")

# %%
original_stdout = sys.stdout
with open("Model_Reports.txt", 'a') as f:
    sys.stdout = f
    print("*****Random Forest Report******\n")
    print("Confusion Matrix:")
    print(matrix)
    print("Classification Report:")
    print(report)
    sys.stdout = original_stdout

# %% train model with all the data
RF = RandomForestClassifier(n_estimators=1000)
RF.fit(X, y)

# save model
pickle.dump(RF, open('RF_model.h5', 'wb'))
