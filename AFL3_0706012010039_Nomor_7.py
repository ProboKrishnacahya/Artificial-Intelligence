# -*- coding: utf-8 -*-
"""# Nomor 7"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

fuel = pd.read_csv('FuelConsumptionCo2.csv')
fuel

fuel = fuel[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
fuel

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    fuel[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']], fuel['CO2EMISSIONS'], test_size=0.2, random_state=42
)

X_train

lr = LinearRegression()
lr.fit(X_train, y_train)

lr.coef_

lr.intercept_

"""y = x1.b1 + x2.b2 + x3.b3 + b0"""

X_train.head(3)

y_train.head(3)

2 * lr.coef_[0] + 4 * lr.coef_[1] + 11.6 * lr.coef_[2] + lr.intercept_

lr.score(X_train, y_train)

pred = lr.predict(X_train)
pred

y_train

from sklearn.metrics import mean_squared_error

mseValue = mean_squared_error(y_train, pred, squared=True)
mseValue
print(f'MSE value on train set: {mseValue}')

testPred = lr.predict(X_test)

mseValue = mean_squared_error(y_test, testPred)
mseValue
print(f'MSE value on test set: {mseValue}')

"""Calculate the predicted emission when ENGINESIZE = 2, CYLINDERS = 5, and FUELCONSUMTION_COMB = 3 (10 points)"""

pred = 2 * lr.coef_[0] + 5 * lr.coef_[1] + 3 * lr.coef_[2] + lr.intercept_

print(f'Predicted Emission: {pred}')