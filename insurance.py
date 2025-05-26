import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("insurance_data.csv")
df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("R-squared:", round(r2, 3))
print("RMSE:", round(rmse, 2))

coefficients = pd.Series(model.coef_, index=X.columns)
print("\nFeature Impact on Charges:")
print(coefficients.sort_values(ascending=False))
