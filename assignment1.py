import statsmodels.formula.api as smf
import pandas as pd

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

#explore the data
data.head()

#Select the data, - assign 1
model = smf.ols("trips ~ hour", data=data)

modelFit = model.fit()

modelFit.summary()
print(modelFit.summary())

#use the test_data
test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
test_data = test_data[['hour']]
test_data.head(10)

#Predict with the test data if it fits.
#(See source github open public model fitting, for not work adjust data input)
pred = modelFit.predict(test_data)

print(pred)