import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)

print("Raw data gender distribution:\n", data['Sex'].value_counts(normalize=True).round(2))

X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def proportional_filtering(df, age_threshold=4, chol_threshold=0):
   
    age_dist = df['Age'].value_counts(normalize=True).round(2)
    df_filtered = df[df['Age'] > age_threshold]

   
    chol_dist = df['HighChol'].value_counts(normalize=True).round(2)
    df_filtered = df_filtered[df_filtered['HighChol'] > chol_threshold]

    return df_filtered

X_train_filtered = proportional_filtering(X_train)
y_train_filtered = y_train.loc[X_train_filtered.index]

X_test_filtered = proportional_filtering(X_test)
y_test_filtered = y_test.loc[X_test_filtered.index]

print("Filtered test set gender distribution:\n", X_test_filtered['Sex'].value_counts(normalize=True).round(2))

encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train_filtered.select_dtypes(include=['object']))
X_test_encoded = encoder.transform(X_test_filtered.select_dtypes(include=['object']))

X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(X_train_filtered.select_dtypes(include=['object']).columns))
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(X_test_filtered.select_dtypes(include=['object']).columns))

X_train_final = pd.concat([X_train_filtered.select_dtypes(exclude=['object']).reset_index(drop=True), X_train_encoded_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_filtered.select_dtypes(exclude=['object']).reset_index(drop=True), X_test_encoded_df.reset_index(drop=True)], axis=1)

X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train_filtered)

y_pred = model.predict(X_test_final)
print(classification_report(y_test_filtered, y_pred))