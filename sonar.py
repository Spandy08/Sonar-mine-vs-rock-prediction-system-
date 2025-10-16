import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sonar_data = pd.read_csv("https://raw.githubusercontent.com/ChandanaGiridhar/Rock_vs_mine_prediction/main/sonar_data.csv", header = None)
print(sonar_data.head())
print(sonar_data.shape)
print(sonar_data.describe) 
print(sonar_data[60].value_counts())
print(sonar_data.groupby(60).mean())
X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60]
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)
print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('accuracy of training data : ', training_data_accuracy)
X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, Y_test)
print('accuracy of training data : ', testing_accuracy) 
input_data = (0.0519, 0.0548, 0.0842, 0.0319, 0.1158, 0.0922,	0.1027,	0.0613,	0.1465,	0.2838,	0.2802,	0.3086,	0.2657,	0.3801,	0.5626,	0.4376,	0.2617,	0.1199,	0.6676,	0.9402,	0.7832,	0.5352,	0.6809,	0.9174,	0.7613,	0.8220,	0.8872,	0.6091,	0.2967,	0.1103,	0.1318,	0.0624,	0.0990,	0.4006,	0.3666,	0.1050,	0.1915,	0.3930,	0.4288,	0.2546,	0.1151,	0.2196,	0.1879,	0.1437,	0.2146,	0.2360,	0.1125,	0.0254,	0.0285,	0.0178,	0.0052,	0.0081,	0.0120,	0.0045,	0.0121,	0.0097,	0.0085,	0.0047,	0.0048,	0.0053)
input_array = np.asarray(input_data)
input_reshape = input_array.reshape(1,-1)
prediction = model.predict(input_reshape)
print(prediction)
if(prediction == 'R') :
    print('The object is a rock')
else : 
    print('The object is a mine')
       





