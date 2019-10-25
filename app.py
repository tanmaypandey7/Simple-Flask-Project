# Import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)  # Load the model
model = pickle.load(open(os.path.join(THIS_FOLDER, 'model/model1.pkl')))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    int_features = [int(x) for x in request.form.values()]
    final_features = int_features
    prediction = model.predict([np.array(final_features)])    
    output = round(prediction[0])
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
