# %% import statements
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from confusion_matrix_plot import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
import sys
import pickle

# %% load dataset
df = pd.read_excel('bankloan_5000.xlsx', sheet_name=0)
print(df)

# get training df
train_df = df[:4000]
test_df = df[4000:]

# %% describe dataframe of only train data
# display all columns
pd.set_option('display.max_columns', None)

print(train_df.describe())

# reset columns to display
pd.reset_option('max_columns')

# %% pandas profiling to generate detailed report for the dataset
# generate report only for train data
report = pandas_profiling.ProfileReport(train_df)
report.to_file("Train_data_bank_loans_report.html")
"""
We observe from the report that:
 1. there no null values in the training set of the first 4000
 2. most borrowers are those in the lower education levels
 3. household income, credit card debt and other debt are highly positively correlated
 4. Percentage of the ratio of debt to income is positively correlated to some extent with default
 
 ....among other observations
However, we observe in the training dataset that there are more than double defaulters than non-defaulters which should 
be an area of concern.
"""

# %% data analysis and visualization
ed_default_df = pd.DataFrame(train_df[['ed', 'default']].value_counts()).sort_values('ed')
# plot all values
ed_default_df.plot(kind='bar', legend=None, title='Education Level Frequencies')
plt.ylabel('Count')
plt.savefig("Education Level Frequencies.png", bbox_inches='tight')
plt.show()
"""
We observe that defaulters are more than non-defaulters in each Education Level
"""

# plot defaulters
train_df[['ed', 'default']][train_df['default'] == 0].value_counts().plot(kind='bar', color='r', title='Education '
                                                                                                       'Level and '
                                                                                                       'Defaulters '
                                                                                                       'Frequencies')
plt.ylabel('Count')
plt.savefig("Education Level and Defaulters Frequencies.png", bbox_inches='tight')
plt.show()
"""
We observe that defaulters decrease as the Education Level increases
"""

# plot non-defaulters
train_df[['ed', 'default']][train_df['default'] == 1].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.title('Education Level and Non-Default Frequencies')
plt.savefig("Education Level and Non-Default Frequencies.png", bbox_inches='tight')
plt.show()
"""
We observe that non-defaulters also decrease as the Education Level increases
"""

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

# normalize data
X = preprocessing.StandardScaler().fit_transform(X)
X = pd.DataFrame(X, columns=cols)
print(X.head())

# %% split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# %% Modeling
# use balanced class weight due to the higher number of defaulters compared to non-defaulters
LR = LogisticRegression(solver='liblinear', class_weight='balanced')
LR.fit(X_train, y_train)
print(LR)

# predict
y_pred = LR.predict(X_test)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)


# %%
plot_confusion_matrix(matrix, classes=['default=0', 'default=1'], normalize=True, title='Logistic Regression '
                                                                                        'Confusion matrix')
plt.savefig("Logistic Regression Confusion matrix.png", bbox_inches='tight')
plt.show()
print(f"Confusion Matrix:\n{matrix}")
print(f"Classification Report:\n{report}")

# %%
original_stdout = sys.stdout
with open("Model_Reports.txt", 'a') as f:
    sys.stdout = f
    print("*****Logistic Regression Report******\n")
    print("Confusion Matrix:")
    print(matrix)
    print("Classification Report:")
    print(report)
    sys.stdout = original_stdout

# %% train model with all the data
LR = LogisticRegression(solver='liblinear', class_weight='balanced')
LR.fit(X, y)

# save model
pickle.dump(LR, open('LR_model.h5', 'wb'))
