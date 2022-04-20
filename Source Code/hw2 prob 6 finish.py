'''
References:
    https://www.ritchieng.com/machine-learning-evaluate-classification-model/
    https://datatofish.com/logistic-regression-python/
    https://stackabuse.com/gradient-descent-in-python-implementation-and-theory/
'''

from sklearn.linear_model import LogisticRegression
import pandas as pd

#import data
x_train = pd.read_csv(r'C:/Users/Berto/data for homework 3/Problem 6_data_trainx.csv')
y_train = pd.read_csv(r'C:/Users/Berto/data for homework 3/Problem 6_data_trainy.csv')

x_test = pd.read_csv(r'C:/Users/Berto/data for homework 3/Problem 6_data_testx.csv')
y_test = pd.read_csv(r'C:/Users/Berto/data for homework 3/Problem 6_data_testy.csv')

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

'''
#visualize decision boundary
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(clf, X_train, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

#determining strength of the regularization called C --> a higher values of c corresponds to
#less regularization. in other words using a high value for parameter C, the functions try to
#fit the training set as best as possible, while with low values of parameter C, the model puts
#more emphasis on finding a coefficient vector (w) that is close to zero
mglearn.plots.plot_linear_svc_regularization()
'''

#accuracy = opposite of error --> 1 - accuracy = error
#here C=1 is default --> C=100 equals higher accuracy while C=0.01 equals lower accuracy
logreg = LogisticRegression(C=1).fit(X_train, y_train)
print("Training set error rate: {:.3f}".format(1 - logreg.score(X_train, y_train)))
print("Test set error rate    : {:.3f}".format(1 - logreg.score(X_test, y_test)))

'''
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score, C=100 : {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score, C=100     : {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score, C=0.01 : {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score, C=0.01     : {:.3f}".format(logreg001.score(X_test, y_test)))
'''

print("weights      : {}".format(logreg.coef_))
print("intercept (b): {}".format(logreg.intercept_))

