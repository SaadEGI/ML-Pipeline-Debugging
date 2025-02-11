import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

def identify_proxy_attributes(df, protected_attributes, correlation_threshold=0.8):
    proxy_attributes = set()
    
   
    df_encoded = pd.get_dummies(df, drop_first=True)
    
   
    corr_matrix = df_encoded.corr().abs()
    
   
    for protected_attribute in protected_attributes:
        if protected_attribute in df.columns:
            encoded_columns = [col for col in df_encoded.columns if col.startswith(protected_attribute)]
            for encoded_col in encoded_columns:
                high_corr_attributes = corr_matrix[encoded_col][corr_matrix[encoded_col] > correlation_threshold].index.tolist()
                proxy_attributes.update(high_corr_attributes)
    
   
    proxy_attributes = {col for col in proxy_attributes if not any(col.startswith(protected_attr) for protected_attr in protected_attributes)}
    
    return list(proxy_attributes)

protected_attributes = ['race', 'gender']

X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis=1), data['salary'], test_size=0.2, random_state=42)

proxy_attributes = identify_proxy_attributes(X_train, protected_attributes)

features_to_include = [col for col in X_train.columns if col not in protected_attributes + proxy_attributes]
X_train = X_train[features_to_include]
X_test = X_test[features_to_include]

categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print(f"Removed proxy attributes: {proxy_attributes}")