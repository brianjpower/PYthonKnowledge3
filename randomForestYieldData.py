import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df_inline = pd.read_csv("inline_data_50lots_new.zip")
df_eol = pd.read_csv("eol_data_50lots_new.zip")

# Merge datasets
data = pd.merge(df_inline, df_eol, on="Device_ID")

# Data Cleaning: Impute missing values with median
#imputer = SimpleImputer(strategy='median')
#data.iloc[:, 1:] = imputer.fit_transform(data.iloc[:, 1:])

# Data Cleaning: Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=["number"]).columns
non_numeric_cols = data.select_dtypes(exclude=["number"]).columns

# Impute missing values in numeric columns with median
imputer = SimpleImputer(strategy="median")
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])


# Feature Selection: Find most important features
X = data.drop(columns=["Wafer_ID","Device_ID","X_x","Y_x","X_y","Y_y","Result", "Fail_Bin"])
y = data["Result"].map({"Pass": 0, "Fail": 1})
feature_importances = mutual_info_classif(X, y, discrete_features=False)
important_features = X.columns[np.argsort(feature_importances)[-10:]]

# Filter dataset to include only important features
X_filtered = X[important_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# Train ML model (Random Forest)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions
y_pred = model_rf.predict(X_test)

# Evaluate performance
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model_rf, "rf_model.pkl")

# Function to predict new inline data
def predict_eol(new_data_path, model_path="rf_model.pkl"):
    model = joblib.load(model_path)
    new_data = pd.read_csv(new_data_path)
    new_data = new_data[important_features]
    new_data = imputer.transform(new_data)
    predictions = model.predict(new_data)
    return predictions

# Example usage
# predictions = predict_eol("new_inline_data.csv")
# print(predictions)
