import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv("C:\\Users\\Sakshi\\Desktop\\EDUREKA\\StudentsPerformance.csv")

# Create a new column for total score
data['total score'] = data['math score'] + data['reading score'] + data['writing score']

# Define the passing criterion: pass if total score >= 210, else fail
data['pass/fail'] = data['total score'].apply(lambda x: 'Pass' if x >= 210 else 'Fail')

# Define features and target variables
X = data.drop(columns=['total score', 'pass/fail'])
y_total_score = data['total score']
y_pass_fail = data['pass/fail']

# Define categorical features
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# One-Hot Encoding for categorical variables
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Split the data into training and test sets
X_train, X_test, y_train_total, y_test_total = train_test_split(X, y_total_score, test_size=0.2, random_state=42)
_, _, y_train_pass_fail, y_test_pass_fail = train_test_split(X, y_pass_fail, test_size=0.2, random_state=42)

# Create pipelines for both regression (total score) and classification (pass/fail)
regressor_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

classifier_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Train the models
regressor_pipeline.fit(X_train, y_train_total)
classifier_pipeline.fit(X_train, y_train_pass_fail)

# Function to predict student performance
def predict_student_performance(input_data):
    predicted_total_score = regressor_pipeline.predict(input_data)
    predicted_pass_fail = classifier_pipeline.predict(input_data)
    return predicted_total_score[0], predicted_pass_fail[0]

# Streamlit UI
st.title("Student Performance Predictor")

st.header("Enter student details")

# Sidebar inputs
gender = st.selectbox('Gender', ['male', 'female'])
race_ethnicity = st.selectbox('Race/Ethnicity', ['group a', 'group b', 'group c', 'group d', 'group e'])
parental_education = st.selectbox("Parental level of education", 
                                  ['bachelor\'s degree', 'some college', 'master\'s degree', 
                                   'associate\'s degree', 'some high school', 'high school'])
lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])
test_prep_course = st.selectbox('Test preparation course', ['none', 'completed'])
math_score = st.slider('Math Score (0-100)', 0, 100, 75)
reading_score = st.slider('Reading Score (0-100)', 0, 100, 75)
writing_score = st.slider('Writing Score (0-100)', 0, 100, 75)

# Create input dataframe
input_data = pd.DataFrame({
    'gender': [gender],
    'race/ethnicity': [race_ethnicity],
    'parental level of education': [parental_education],
    'lunch': [lunch],
    'test preparation course': [test_prep_course],
    'math score': [math_score],
    'reading score': [reading_score],
    'writing score': [writing_score]
})

# Predict button
if st.button('Predict'):
    total_score, pass_fail = predict_student_performance(input_data)
    
    st.write(f"### Predicted Total Score: {total_score:.2f}")
    st.write(f"### Predicted Result: {pass_fail}")

