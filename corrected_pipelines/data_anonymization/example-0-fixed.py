import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

def anonymize_data(df):
    df['education'] = df['education'].apply(lambda x: 'anon' if x in ['Doctorate', 'Masters'] else x)
    return df

data = anonymize_data(data)

le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

X = data.drop(columns=['salary'])
y = data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DP_RandomForestClassifier(random_state=42, epsilon=1.0) 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the corrected pipeline: {accuracy}')
print(f'Classification report: {classification_report(y_test, y_pred)}')