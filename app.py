import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('et.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    #return render_template('result.html', prediction_text='Your Rating is: {}'.format(output))
    return render_template('result.html', output=f"Predicted Rating is: {str(output)}")

if __name__ == "__main__":
    app.run(debug=True)