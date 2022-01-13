import sklearn.model_selection
from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("housePrice.csv", sep=",")  # read the dataset

data = data[["Area", "Room", "Parking", "Warehouse", "Elevator", "Address", "Price", "Price(USD)"]] # choose some data that we want

# with this loop we can change the number of Area from str to int
num = 0
for x in data["Area"]:
    k = x
    if type(x) is str:
        k = ''.join([n for n in x if n.isdigit()])
    data["Area"][num] = int(k)

    if int(k) >= 1000:  # if the Area is bigger that 1000 remove it
        data.drop(num, axis=0, inplace=True)

    num += 1

data = data[data["Address"].notna()]  # remove the rows that its Address is empty

# change string class label to int
addr = preprocessing.LabelEncoder()
addr.fit(data.Address)
data['Address'] = addr.transform(data.Address)

# these two lines will remove the empty rows and then store it again
data.dropna(axis=0, how='all', inplace=True)
data.to_csv('housePrice.csv', index=False)


predict = "Price"  # define the predict value

x = np.array(data.drop([predict], 1))  # choose the x values
y = np.array(data[predict])  # choose the y value for predict

# divide the data to train/test and split the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
linear = linear_model.LinearRegression()  # create an instance from linear model
linear.fit(x_train, y_train)  # find the fit line

result = linear.predict(x_test)  # predict the result with test data
for x in range(len(result)):  # show the result, original price and predicted price
    print("With these data : ", x_test[x], ", The predicted price is : ", result[x], "And the original price is : ", y_test[x])
