from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pymysql.cursors
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

app = Flask(__name__)

# Function to connect to the MySQL Database
def connect_to_database(host, user, password, database):
    try:
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database,
                                     cursorclass=pymysql.cursors.DictCursor)
        print("Connected to the database successfully!")
        return connection
    except pymysql.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

# Load your dataset
data = pd.read_csv('/users/shridayalyadav/downloads/az.csv')

# MySQL Database credentials
host = 'localhost'
user = 'root'
password = 'data@1234'
database = 'college_prediction'

# Connect to the database
connection = connect_to_database(host, user, password, database)

# Split the dataset into features (X) and target variable (y)
X = data[['12th_score', 'MH_CET_score', 'JEE_score']]
y = data['Admission_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines for preprocessing and training the models
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear'))
])

naive_bayes_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('naive_bayes', GaussianNB())
])

# Define hyperparameters grid for SVM
svm_param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto'],
}

# Define hyperparameters grid for Naive Bayes
naive_bayes_param_grid = {
    # No hyperparameters to tune for Gaussian Naive Bayes
}

# Perform hyperparameter tuning using GridSearchCV for SVM
svm_grid_search = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, n_jobs=-1)
svm_grid_search.fit(X_train, y_train)
best_svm_model = svm_grid_search.best_estimator_

# Perform hyperparameter tuning using GridSearchCV for Naive Bayes
naive_bayes_grid_search = GridSearchCV(naive_bayes_pipeline, naive_bayes_param_grid, cv=5, n_jobs=-1)
naive_bayes_grid_search.fit(X_train, y_train)
best_naive_bayes_model = naive_bayes_grid_search.best_estimator_

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Parse request data
        data = request.json
        print(data)
        try:
            print("here")
            twelfth_score = data['12th_score']
            exam_choice = data['exam_choice']
            exam_score = data[exam_choice] if exam_choice in data else 0
        except KeyError as e:
            return jsonify({'error': f'Missing key in JSON data: {str(e)}'}), 400

        # Make predictions based on exam choice
        if exam_choice == 'MH_CET_score':
            X_pred = [[twelfth_score, exam_score, 0]]
        elif exam_choice == 'JEE_score':
            X_pred = [[twelfth_score, 0, exam_score]]
        else:
            X_pred = [[twelfth_score, 0, 0]]

        # Predict admission status using best SVM model
        svm_prediction = best_svm_model.predict(X_pred)

        # Predict admission status using best Naive Bayes model
        naive_bayes_prediction = best_naive_bayes_model.predict(X_pred)
        # Predict admission status using C5.0 algorithm
        c50_model = train_c50(X_train, y_train)
        c50_prediction = predict_c50(c50_model, X_pred)
        print(svm_prediction[0], naive_bayes_prediction[0], c50_prediction[0])
        # Prepare and return response
        response = {
            'SVM_Prediction': svm_prediction[0],
            'Naive_Bayes_Prediction': naive_bayes_prediction[0],
            'C50_Prediction': c50_prediction[0]
        }
        
         # Additional: Model Evaluation
        svm_prediction_test = best_svm_model.predict(X_test)
        naive_bayes_prediction_test = best_naive_bayes_model.predict(X_test)
        c50_prediction_test = predict_c50(c50_model, X_test)
        evaluation = {
            'SVM_Test_Report': classification_report(y_test, svm_prediction_test),
            'Naive_Bayes_Test_Report': classification_report(y_test, naive_bayes_prediction_test),
            'C50_Test_Report': classification_report(y_test, c50_prediction_test)
        }
        
        response.update(evaluation)

        return jsonify(response)

# Route to render the result.html template
@app.route('/result')
def result():
    return render_template('result.html', svm_prediction="SVM_Prediction", naive_bayes_prediction="Naive_Bayes_Prediction", c50_prediction="C50_Prediction")

if __name__ == '__main__':
    app.run(debug=True)

# Close the database connection
if connection:
    connection.close()

def train_c50(X_train, y_train):
    robjects.r('library(C50)')
    c50_model = robjects.r('C5.0')
    X_train_r = pandas2ri.py2rpy(X_train)
    y_train_r = pandas2ri.py2rpy(y_train)
    model = c50_model(X_train_r, y_train_r)
    return model

def predict_c50(model, X_pred):
    X_pred_r = pandas2ri.py2rpy(pd.DataFrame(X_pred, columns=['12th_score', 'MH_CET_score', 'JEE_score']))
    predictions = robjects.r.predict(model, newdata=X_pred_r)
    return list(predictions)


