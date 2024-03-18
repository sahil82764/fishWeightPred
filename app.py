from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load the dataset
fish_data = pd.read_csv("Fish.csv")

# Define features and target
X = fish_data[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = fish_data['Weight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])
        

        # Make prediction using the model
        prediction = model.predict([[length1, length2, length3, height, width]])

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)