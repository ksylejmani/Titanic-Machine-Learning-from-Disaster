import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train_file_path = 'input/train.csv'
test_file_path = 'input/test.csv'
X = pd.read_csv(train_file_path)
X_test = pd.read_csv(test_file_path)

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['Survived'], inplace=True)
y = X.Survived
X.drop(['Survived'], axis=1, inplace=True)

categorical_columns = [col for col in X if X[col].dtype == 'object']
numerical_columns = [col for col in X if X[col].dtype in ['int32', 'int64', 'float64']]

# Preprocessor for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessor for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessor for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Create the model
model = RandomForestClassifier(n_estimators=30000, random_state=0)
# Bundle preprocessor and modeling code in a pipeline
titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model
titanic_pipeline.fit(X_train, y_train)

# Predict for validation data
predictions_valid = titanic_pipeline.predict(X_valid)

# Evaluate the model
score = roc_auc_score(y_valid, predictions_valid)
print("roc_auc_score: ", score)

# Make predictions using test data for submission to competition
predictions = titanic_pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': predictions})
output.to_csv('output/submission.csv', index=False)
print("Predictions are saved to file")
