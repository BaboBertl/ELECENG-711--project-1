import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn.neural_network import MLPClassifier

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

#simulation to determine best neural network configuration
fig, axes = plt.subplots(2, 4, figsize=(20, 8)) 
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='adam', max_iter=4000, random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))

#creating neural network
mlp1 = MLPClassifier(solver='adam', max_iter=4000, random_state=0, hidden_layer_sizes=[100, 100], alpha=0.01).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp1, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#display answer
print("error rate on training set: {:.2f}".format(1 - mlp1.score(X_train, y_train)))
print("error rate on test set: {:.2f}".format(1 - mlp1.score(X_test, y_test)))

