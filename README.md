import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# Fixed: changed AveragePrecisionScore to average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
def load_data():
    """
    Attempts to load the standard Kaggle credit card fraud dataset.
    Falls back to a synthetic dataset if the file is not found.
    """
    try:
        # Replace 'creditcard.csv' with your actual file path if needed
        df = pd.read_csv('creditcard.csv')
        print("Successfully loaded 'creditcard.csv'")
    except FileNotFoundError:
        print("Warning: 'creditcard.csv' not found. Generating synthetic fraud data for demonstration...")
        # Generating synthetic data mimicking the real dataset structure
        np.random.seed(42)
        n_samples = 50000
        n_features = 28
        
        # 99.8% genuine, 0.2% fraud
        frauds = int(n_samples * 0.002) 
        
        X_syn = np.random.randn(n_samples, n_features)
        # Shift fraud data slightly to make it distinguishable but challenging
        X_syn[:frauds] += 1.5 
        
        amounts = np.random.exponential(scale=100, size=(n_samples, 1))
        times = np.linspace(0, 172800, n_samples).reshape(-1, 1)
        
        data = np.hstack([times, X_syn, amounts])
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        df = pd.DataFrame(data, columns=columns)
        df['Class'] = 0
        df.iloc[:frauds, -1] = 1 # Set the first few as fraud
        
    return df

df = load_data()

# ==========================================
# 2. PREPROCESSING & NORMALIZATION
# ==========================================
# The 'Time' and 'Amount' features need scaling. V1-V28 are usually already PCA-scaled.
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original unscaled columns
df = df.drop(['Time', 'Amount'], axis=1)

# Separate features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Fixed: Added missing quotes around the string
print("\nClass Distribution:")
print(y.value_counts(normalize=True))

# ==========================================
# 3. TRAIN-TEST SPLIT (Stratified)
# ==========================================
# We use stratify=y to ensure both train and test sets have the same ratio of fraud
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. HANDLING CLASS IMBALANCE (SMOTE)
# ==========================================
print(f"\nBefore SMOTE - Training target shape: {np.bincount(y_train)}")

# Oversample the minority class (fraud) only on the TRAINING set to avoid data leakage
smote = SMOTE(sampling_strategy=0.1, random_state=42) # Brings fraud up to 10% of majority class
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training target shape: {np.bincount(y_train_res)}")

# ==========================================
# 5. MODEL TRAINING (Random Forest)
# ==========================================
# Fixed: Added missing quotes around the string
print("\nTraining Random Forest Classifier (this may take a moment)...")
# class_weight='balanced_subsample' adds an extra layer of protection against imbalance
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# ==========================================
# 6. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("             MODEL EVALUATION REPORT            ")
print("="*50)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Genuine caught): {cm[0][0]}")
print(f"False Positives (False Alarms):  {cm[0][1]}")
print(f"False Negatives (Missed Fraud):  {cm[1][0]}")
print(f"True Positives (Fraud caught):   {cm[1][1]}")

# Classification Report (Precision, Recall, F1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))

# Fixed: Changed to lowercase average_precision_score
auprc = average_precision_score(y_test, y_probs)
print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
