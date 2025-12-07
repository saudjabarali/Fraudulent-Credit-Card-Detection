# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Load the dataset
df = pd.read_csv("C:\Users\thesa\OneDrive\Documents\SKILLIFIED MENTOR\Project recognize fraudulent credit card transactions\Project recognize fraudulent credit card transactions Final Report\creditcard.csv")
df.info()
df.head()

# Count of fraud vs non-fraud
plt.figure(figsize=(6,4))
sns.countplot(x='Class', hue='Class', data=df, palette='Set2', legend=False)
plt.title('Count of Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.show()

# Print counts too
print("Class distribution:\n", df['Class'].value_counts())

# Correlation heatmap
corr_matrix = df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(
    corr_matrix[['Class']].sort_values(by='Class', ascending=False),
    annot=True, cmap='coolwarm', fmt=".2f"
)
plt.title('Correlation of Features with Fraudulent Class')
plt.show()



# Feature scaling for 'Amount' and 'Time'
# Make a copy to avoid touching original dataframe
df_scaled = df.copy()

# Apply standard scaling
scaler = StandardScaler()
df_scaled[['scaled_amount', 'scaled_time']] = scaler.fit_transform(df_scaled[['Amount', 'Time']])


# Drop original columns
df_scaled.drop(['Amount', 'Time'], axis=1, inplace=True)


# Rearranging columns# Optional: Rearranging for better readability
columns = [col for col in df_scaled.columns if col != 'Class'] + ['Class']
df_scaled = df_scaled[columns]

# Optional: Rearranging for better readability
columns = [col for col in df_scaled.columns if col != 'Class'] + ['Class']
df_scaled = df_scaled[columns]

# Define features and target
X = df_scaled.drop('Class', axis=1)
y = df_scaled['Class']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

# Confusion matrix and classification report


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# Accuracy score
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

