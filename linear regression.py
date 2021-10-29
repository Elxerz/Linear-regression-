import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
X = np.array(data.drop([predict], 1))# return a new data without G3
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''best = 0
for i in range(25):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #splitting up 10% of
    # our of data into test samples so that when we test we can test off that and its never seen that informaton before to avoid
    #inaccurate result

    # Creating a model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)# fit this data to find a best fit line
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:  # save the model
            pickle.dump(linear, f)'''



pickle_in = open('studentmodel.pickle', 'rb') #load the model
linear = pickle.load(pickle_in)


print('Coefficient: \n', linear.coef_)  #
print('Intercept: \n', linear.intercept_)  # Y intercept

predictions = linear.predict(x_test) # predict x_test data without x_train data
for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'studytime'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()








