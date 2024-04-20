# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df = pd.read_csv('IIP_data.csv')
df

# @title Price of goods over time

import matplotlib.pyplot as plt
sns.lineplot(data=df, x="Period", y="Basic")
sns.lineplot(data=df, x="Period", y="Capital")
sns.lineplot(data=df, x="Period", y="Intermediate")
sns.lineplot(data=df, x="Period", y="Consumer total")
sns.lineplot(data=df, x="Period", y="Consumer Durable")
sns.lineplot(data=df, x="Period", y="Consumer Non-durable")
plt.title("Price of goods over time")
plt.xlabel("Period")
_ = plt.ylabel("Price")

df.info()

df.describe()

df.isnull().sum()

import yfinance as yf
import datetime as dt

continuous_cols = [col for col in df.columns if df[col].nunique()>15]
print(continuous_cols)

num_lst = []
cat_lst = []

from pandas.api.types import is_string_dtype, is_numeric_dtype

for column in df:
    plt.figure(column, figsize = (5,5))
    plt.title(column)
    if is_numeric_dtype(df[column]):
        df[column].plot(kind = 'hist')
        num_lst.append(column)
    elif is_string_dtype(df[column]):
        df[column].value_counts().plot(kind = 'bar')
        cat_lst.append(column)

print(num_lst)
print(cat_lst)

import os

print(df.isnull().sum().sum())

X = df.iloc[:, :-1].values #returns all rows and first column
y = df.iloc [:, -1].values #returns all rows and last column

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

specific_row = df.iloc[3]
X = df.iloc[1:,0].values.reshape(-1,1)
y = df.iloc[1:,4].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)
poly_reg = PolynomialFeatures(degree=5)

X_poly = poly_reg.fit_transform(train_X)
X_poly_test = poly_reg.fit_transform(test_X)
model = LinearRegression()

model.fit(X_poly,train_y)
m1 = model.coef_
print("y = {}*Period + {}".format(m1[0][1],m1[0][0]))

# Predict the test set results
y_pred = model.predict(X_poly_test)
mse = mean_squared_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)
print(r2)
print(model.coef_)

plt.figure(figsize=(10,10))
X = df.iloc[1:,0].values
y = df.iloc[1:,2].values
plt.scatter(X,y)
plt.show()

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

print("MSE : ",mean_squared_error(y_pred,test_y))
print("MAE : ",mean_absolute_error(y_pred,test_y))

print(y)

X = df.iloc[1:,0].values.reshape(-1,1)
y = df.iloc[1:,2].values.reshape(-1,1)

y = y.reshape(len(y),1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.xlabel('Months yearwise')
plt.ylabel('IIP of basic goods')
plt.show()



plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('SVR')
plt.xlabel('Months yearwise')
plt.ylabel('IIP of basic goods')
plt.show()


# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[145]])).reshape(-1,1))
