import numpy as np
from sklearn.linear_model import LinearRegression
import math

# This example is taken from "Statistical Investigations" lesson from Maths Standard
x = np.array([[60.2], [61.3], [63.5], [63.7], [65], [66.4], [66.9], [70.1], [70.2], [70.5], [71.6], [72], [72], [72.5], [73.7], [73.8], [74.8], [75.9], [76.3], [78.9]])
y = np.array([88.1, 91.5, 88.7, 90.6, 90.9, 94, 92.1, 96.5, 93.9, 95.2, 94.9, 95.7, 95.1, 96.4, 96.6, 97, 96.3, 99.1, 98.2, 99.9])
model = LinearRegression()
model.fit(x, y)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the equation
print(f"Equation: y = {intercept} + {coefficients[0]} * x")
print(f"Rounded equation: y = {round(intercept,2)} + {round(coefficients[0],2)} * x")

# Get r squared
rSquared = model.score(x, y)
r = math.sqrt(rSquared)
if coefficients[0] < 0:
    r = -r
print(f"Pearsons correlation coefficient, r is {r}. Rounded to 2 dp: {round(r,2)}")


test_weight = 75
test = np.array([test_weight])
y_prediction = model.predict(test.reshape(1, -1))
print(f"predicted glucose level for {test_weight}kg is: {y_prediction}mg/100mL")

test_weight = 90
test = np.array([test_weight])
y_prediction = model.predict(test.reshape(1, -1))
print(f"predicted glucose level for {test_weight}kg is: {y_prediction}mg/100mL")

