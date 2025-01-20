import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Load the dataset
data = pd.read_csv('oil_spill_synthetic_data.csv')

# Encode the target variable
label_encoder = LabelEncoder()
data['impact_encoded'] = label_encoder.fit_transform(data['impact'])

# Feature selection and target variable
X = data[['spill_size', 'toxicity_level', 'distance_to_coast']]
y = data['impact_encoded']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning for Random Forest
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf_model, param_dist, n_iter=50, scoring='accuracy', cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Use the best estimator
best_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print results
print(f"Best Parameters: {random_search.best_params_}")
print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(report)
