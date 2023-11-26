# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

# importing dataset
exercise = pd.read_csv('exercise.csv')
print(exercise)

calories = pd.read_csv('calories.csv')
print(calories)

# Data Preprocessing

# Taking care of missing values

print(exercise.isnull().values.any())
print(calories.isnull().values.any())

print(exercise.isnull().values.sum())
print(calories.isnull().values.sum())

# combining dataset

calories = pd.concat([exercise, calories['Calories']], axis = 1)

print(calories.head())


# checking the number of rows and cols

print(calories.shape)

# getting some info about data

print(calories.info())

# Data Analysis

# get some staitscal measure about the data
print(calories.describe())


# data visualization

sns.set()
# ploting the gender col in count plot
sns.countplot(data=calories, x='Gender')

# finding the distribution of 'Age' col
sns.displot(calories['Age'])

# finding the distribution of 'Height' col
sns.displot(calories['Height'])

# finding the distribution of 'weight' col
sns.displot(calories['Weight'])

# finding the correlation in the dataset
'''1. positve correlation
   2. negative correlation'''

correlation = calories.corr()

# constructing heatmap to understand the correlation

plt.figure(figsize=(10, 10))
sns.heatmap(correlation, annot=True, cmap='Blues', square= True, fmt=".1f", linewidths=0.5)


# encoding categorical data
calories.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

print(calories)

# Features variables( Independent variables and Dependent Varibles)

X = calories.iloc[:, :-1].values
print(X)
Y = calories.iloc[:, 8:].values
print(Y)

# spliting the dataset into training data set and test data set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

# Model Training

# train the model using training data set

# train the model using train data set
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# predict the results for X_test
y_pred = regressor.predict(X_test)
print(y_pred)

print(Y_test)

# accuracy measure

mae = metrics.mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Aboslute Error : {mae}")
print("Mean Squared Error:", mse)
print("R-squared (R²):", r2)

# visualizing

indices = np.arange(len(Y_test))

plt.plot(indices, Y_test, label='Actual Values', marker='o')
plt.plot(indices, y_pred, label='Predicted Values', marker='x')
plt.xlabel("Instances")
plt.ylabel("Values")
plt.title("Comparison between Y_test and y_pred")
plt.legend()
plt.show()




# You can Skip This 

# Assuming X_train, Y_train are defined
# Convert Y_train to a DataFrame
Y_train_df = pd.DataFrame({'Target': Y_train.flatten()})

# Combine X_train and Y_train into a single DataFrame
train_data = pd.concat([pd.DataFrame(X_train), Y_train_df], axis=1)

# Plot pair plot with titles and labels
sns.set(style="whitegrid")
sns.pairplot(train_data, hue='Target')
plt.suptitle('Pair Plot of Training Data', y=1.02)
plt.show()
