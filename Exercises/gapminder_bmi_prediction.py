'''
The data file can be found under the "bmi_and_life_expectancy.csv" tab in the quiz below. It includes three columns, containing the following data:

Country – The country the person was born in.
Life expectancy – The average life expectancy at birth for a person in that country.
BMI – The mean BMI of males in that country.
You'll need to complete each of the following steps:
1. Load the data

The data is in the file called "bmi_and_life_expectancy.csv".
Use pandas read_csv to load the data into a dataframe (don't forget to import pandas!)
Assign the dataframe to the variable bmi_life_data.
2. Build a linear regression model

Create a regression model using scikit-learn's LinearRegression and assign it to bmi_life_model.
Fit the model to the data.
3. Predict using the model

Predict using a BMI of 21.07931 and assign it to the variable laos_life_exp.
'''

# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('Excercises\\1bmi_and_life_expectancy.csv')
print(bmi_life_data.head(10)) 

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])

print('The life expectancy for a person with BMI : 21.07931 is {}'.format(laos_life_exp))

laos_life_exp = bmi_life_model.predict([[18.07931]])

print('The life expectancy for a person with BMI : 18.07931 is {}'.format(laos_life_exp))