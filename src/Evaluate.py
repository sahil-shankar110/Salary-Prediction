import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv("Data/Salary_Data_Cleaned.csv")

X = df[["Age","Gender_Encoded","Education_Level_Encoded","Job_Title_Encoded","Years of Experience"]]
y = df["Salary"]

model = joblib.load("Models/salary_model.pkl")
y_pred = model.predict(X)

print("R2 Score:", r2_score(y, y_pred))
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
