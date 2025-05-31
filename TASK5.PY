
# TASK 5: SALES PREDICTION USING PYTHON

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'ad_budget': [1000, 2000, 3000, 4000],
    'sales': [10, 20, 30, 40]
}
df = pd.DataFrame(data)

# Features and label
X = df[['ad_budget']]
y = df['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Predicted Sales:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))

# Input: advertising budget (₹)
# Output: predicted sales (units)

