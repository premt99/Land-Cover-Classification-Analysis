import pandas as pd
import json
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import os

# Load Dataset
file_path = input("Enter the full path to the Maharashtra dataset file (e.g., 'Maharashtra.csv'): ").strip()
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' was not found.")

df = pd.read_csv(file_path)

# Parse `.geo` column to extract longitude and latitude
if '.geo' in df.columns:
    df['longitude'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][0])
    df['latitude'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][1])

# Drop rows with missing coordinates
df.dropna(subset=['longitude', 'latitude'], inplace=True)

# Add a synthetic year column for simulation (assuming data spans from 2017 to 2024)
num_years = 2024 - 2017 + 1
df['year'] = pd.Series(np.tile(np.arange(2017, 2017 + num_years), int(np.ceil(len(df) / num_years)))[:len(df)])

# Define central point for Maharashtra (Mumbai)
central_point = (19.0760, 72.8777)
df['distance_from_center'] = df.apply(
    lambda row: geodesic((row['latitude'], row['longitude']), central_point).kilometers
    if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) else None,
    axis=1
)
df['lon_lat_interaction'] = df['longitude'] * df['latitude']

# Define feature set and target variable
if 'classification' not in df.columns:
    raise ValueError("The dataset must contain a 'classification' column for target labels.")
X = df[['longitude', 'latitude', 'distance_from_center', 'lon_lat_interaction']]
y = df['classification']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train models and store accuracies
model_accuracies = {}

# Gradient Boosting
gb_classifier = GradientBoostingClassifier()
param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'subsample': [0.8, 1.0]}
gb_grid = GridSearchCV(gb_classifier, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_
gb_accuracy = accuracy_score(y_test, gb_best.predict(X_test))
model_accuracies['Gradient Boosting'] = gb_accuracy

# Random Forest
rf_classifier = RandomForestClassifier(class_weight='balanced')
param_grid_rf = {'n_estimators': [100, 150], 'max_depth': [10, None], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_accuracy = accuracy_score(y_test, rf_best.predict(X_test))
model_accuracies['Random Forest'] = rf_accuracy

# Decision Tree
dt_classifier = DecisionTreeClassifier(max_depth=12, class_weight='balanced')
dt_classifier.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
model_accuracies['Decision Tree'] = dt_accuracy

# Ensemble Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[('gb', gb_best), ('rf', rf_best), ('dt', dt_classifier)],
    voting='soft'
)
voting_classifier.fit(X_train, y_train)
voting_accuracy = accuracy_score(y_test, voting_classifier.predict(X_test))
model_accuracies['Voting Classifier'] = voting_accuracy

# Display Model Accuracy Metrics
print("\nModel Accuracies:")
for model, accuracy in model_accuracies.items():
    print(f"{model} Accuracy: {accuracy:.2f}")

# Confusion Matrix for Ensemble Classifier
voting_predictions = voting_classifier.predict(X_test)
voting_cm = confusion_matrix(y_test, voting_predictions)
sns.heatmap(voting_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Ensemble Voting Classifier Confusion Matrix")
plt.show()

# Predict land cover distribution trends for each class across years
future_year = 2030  # Change this to any future year you'd like to predict
forecast_results = {}

for land_type in df['classification'].unique():
    ts = df[df['classification'] == land_type].groupby('year').size()

    if len(ts) > 1:
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.predict(start=ts.index[-1] + 1, end=future_year)
            forecast_results[land_type] = forecast[future_year]
        except Exception as e:
            print(f"ARIMA model error for class {land_type}: {e}")

# Save Predicted Map
prediction_map = folium.Map(location=[19.7515, 75.7139], zoom_start=6)  # Adjust for Maharashtra
prediction_map.save('Predicted_LandCover_Map_Maharashtra.html')
print("Predicted map saved as 'Predicted_LandCover_Map_Maharashtra.html'.")
