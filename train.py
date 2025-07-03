# # train.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import pickle
# from sklearn.preprocessing import StandardScaler

# # Load the crop recommendation dataset
# data = pd.read_csv('Crop_recommendation.csv')  # Ensure this file is in the same directory

# # Separate features and target variable
# X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
# y = data['label']

# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Initialize models
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
#     'Bayes Net': GaussianNB()
# }

# # Train and save each model
# """for name, model in models.items():
#     model.fit(X_train, y_train)
#     with open(f'{name.replace(" ", "_").lower()}_model.pkl', 'wb') as file:
#         pickle.dump(model, file)
# """
# accuracies = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     accuracy = accuracy_score(y_test, model.predict(X_test))
#     accuracies[name] = accuracy
#     # Save model
#     with open(f"{name.replace(' ', '_').lower()}_model.pkl", 'wb') as file:
#         pickle.dump(model, file)

# # Save accuracies
# with open("model_accuracies.pkl", "wb") as file:
#     pickle.dump(accuracies, file)
# print("Models have been trained and saved.")
# train_and_save_models.py
#### this is new
#my
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# === Robust Path Handling ===
BASE_DIR = os.path.dirname(__file__)  # Directory where train.py is located
csv_path = os.path.join(BASE_DIR, 'Crop_recommendation.csv')
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

# === Load the dataset ===
data = pd.read_csv(csv_path)

# === Feature Selection ===
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize Models ===
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'Bayes Net': GaussianNB()
}

# === Standard Scaler for Specific Models ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

accuracies = {}

# === Train Models & Save ===
for name, model in models.items():
    if name in ['Logistic Regression', 'Bayes Net']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy

    model_filename = f"{name.replace(' ', '_').lower()}_model.pkl"
    model_path = os.path.join(models_dir, model_filename)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ {name} trained and saved to {model_path} (Accuracy: {accuracy:.2f})")

# === Save Accuracies ===
with open(os.path.join(models_dir, "model_accuracies.pkl"), "wb") as f:
    pickle.dump(accuracies, f)

# === Save Scaler ===
with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("\n🚀 All models, accuracies, and scaler have been saved successfully.")


