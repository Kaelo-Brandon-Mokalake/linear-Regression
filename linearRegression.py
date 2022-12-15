import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model

# loading the dataset
d = load_diabetes()

# features
d_X = d.data[:, np.newaxis, 2]

# testing data
dx_train = d_X[:-20]
dy_train = d.target[:-20]

# training data
dx_test = d_X[-20:]
dy_test = d.target[-20:]

# creating an object
regr = linear_model.LinearRegression()

# training the model
regr.fit(dx_train, dy_train)

# plot the data to the graph
plt.scatter(dx_test,  dy_test,  color='green', label="Test")
plt.scatter(dx_train,  dy_train,  color='red', label='Training')
plt.plot(dx_test, regr.predict(dx_test), color='blue', linewidth=3, label='Best fit line')

plt.xticks(())
plt.yticks(())
plt.legend()

# displaying data in a graph
plt.show()
print(regr.score(dx_test, dy_test))
