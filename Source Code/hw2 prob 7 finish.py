'''
References:
    https://www.ritchieng.com/machine-learning-evaluate-classification-model/
    https://datatofish.com/logistic-regression-python/
    https://stackabuse.com/gradient-descent-in-python-implementation-and-theory/
'''

from sklearn.linear_model import LogisticRegression
import pandas as pd

#import data
x_train = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_trainx.csv')
y_train = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_trainy.csv')

x_test = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_testx.csv')
y_test = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_testy.csv')

#combine data
frames_train = [x_train, y_train]
df_train = pd.concat(frames_train, axis=1, join='inner')

frames_test = [x_test, y_test]
df_test = pd.concat(frames_test, axis=1, join='inner')

#separate data
X_train = df_train[['x1', 'x2']]
y_train = df_train['y1']

X_test = df_test[['x1', 'x2']]
y_test = df_test['y1']

#convert dataframe to array
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

#display answers
logreg = LogisticRegression(C=1).fit(X_train, y_train)
print("Training set error rate: {:.3f}".format(1 - logreg.score(X_train, y_train)))
print("Test set error rate    : {:.3f}".format(1 - logreg.score(X_test, y_test)))
print("weights      : {}".format(logreg.coef_))
print("intercept (b): {}".format(logreg.intercept_))

