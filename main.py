import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("🎯 Parkinson's Disease Detection using Audio Features")
print("=" * 60)

# Load the data
data_path = "C:\\Users\\prati\\OneDrive\\Desktop\\NeonDungeons\\parkinson\\parkinson-ai\\data\\pd_speech_features\\pd_speech_features.csv"

# Read CSV with proper header handling
# The file has multi-level headers, so we'll use the second row as column names
data_temp = pd.read_csv(data_path, header=None)

# Get feature names from the second row (index 1)
feature_names = data_temp.iloc[1].values
feature_names[-1] = 'class'  # Rename last column to 'class'

# Load data starting from row 2 (index 2) with proper column names
data_clean = pd.read_csv(data_path, skiprows=2, header=None, names=feature_names)

# Convert all columns to numeric
for col in data_clean.columns:
    data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')

print(f"📊 Dataset Overview:")
print(f"   Shape: {data_clean.shape}")
print(f"   Features: {data_clean.shape[1] - 1}")
print(f"   Samples: {data_clean.shape[0]}")

# Display basic information
print(f"\n📈 Class Distribution:")
class_counts = data_clean['class'].value_counts()
print(f"   Healthy (0): {class_counts[0]} ({class_counts[0]/len(data_clean)*100:.1f}%)")
print(f"   Parkinson's (1): {class_counts[1]} ({class_counts[1]/len(data_clean)*100:.1f}%)")

# Check for missing values
print(f"\n🔍 Missing Values: {data_clean.isnull().sum().sum()}")

print("\n" + "=" * 60)

# Data preprocessing
print("🔧 Data Preprocessing...")

# Remove ID and gender columns as they're not useful for prediction
columns_to_drop = [col for col in data_clean.columns if col.lower() in ['id', 'gender'] or col == '0']

data_clean = data_clean.drop(columns=columns_to_drop)

# Separate features and target
X = data_clean.drop('class', axis=1)
y = data_clean['class']

print(f"   Final feature count: {X.shape[1]}")

# Fill any NaN values with median
X = X.fillna(X.median())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Data preprocessing completed!")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

print("\n" + "=" * 60)

# Train Neural Network model
print("🤖 Training Neural Network Model...")

nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
nn_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = nn_model.predict(X_test)
y_pred_proba = nn_model.predict_proba(X_test)[:, 1]

# Calculate test metrics
test_accuracy = nn_model.score(X_test, y_test)
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"🏆 Model: Neural Network")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Test AUC-ROC:  {test_auc:.4f}")

print("\n📊 Detailed Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s']))

print("\n🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"               H    P")
print(f"Actual    H   {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"          P   {cm[1,0]:3d}  {cm[1,1]:3d}")

# Export the trained model
model_path = "C:\\Users\\prati\\OneDrive\\Desktop\\NeonDungeons\\parkinson\\parkinson-ai\\model\\nn_model.pkl"
joblib.dump(nn_model, model_path)
print(f"Model saved to {model_path}")

print("=" * 60)
print("✅ Model training and evaluation completed!")
print("💾 Model exported as .pkl file for deployment.")
print("=" * 60)
