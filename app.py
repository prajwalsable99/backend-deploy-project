from flask import Flask, jsonify ,request
import json
import numpy as np
import joblib
from flask_cors import CORS

with open('data.json', 'r') as f:

    data = json.load(f)


with open('file2_data.json', 'r') as f:

    data2 = json.load(f)

app = Flask(__name__)
CORS(app)


@app.route('/inputdata')
def inputdata():
    try:
        return jsonify({ 'success':True,'statedata':data}) ,200
    except:
        return jsonify({'error': 'something went wrong '}), 500

@app.get('/visualize')
def visualize():
    try:
        return jsonify({ 'success':True,'ricedata':data2}) ,200
    except:
        return jsonify({'error': 'something went wrong '}), 500


model = joblib.load('model.pkl')

@app.post('/predict')
def predict():
    # Get the request data
    try:
        inp_data = request.get_json()

        # Transform the data to match the model's input format
        X = np.array([[inp_data['dcode'], inp_data['year'], inp_data['scode']]])

        # Make the prediction
        prediction = model.predict(X)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist() ,"success":True}),200


    except:
        return jsonify({'error': 'Unable to fetch at moment  '}), 500


rec_model = joblib.load('recommendation_model.pkl')

@app.post('/recommend')
def recommend():
    # Get the request data
    try:
        inp_d = request.get_json()
        inp_=  [inp_d['N'] ,inp_d['P'],inp_d['K'],inp_d['temperature'],inp_d['humidity'],inp_d['ph'],inp_d['rainfall']] 
        # Transform the data to match the model's input format
        X = np.array([inp_])
        
        # Make the prediction
        prediction = rec_model.predict(X)

        print(prediction)

        # Return the prediction as JSON
        return jsonify({'recommendation': prediction[0]  ,"success":True}),200


    except:
        return jsonify({'error': 'Unable to fetch at moment  '}), 500











wheat_m=joblib.load('./wheat.pkl')
sugarcane_m=joblib.load('./sugarcane.pkl')
soyabean_m=joblib.load('./soyabean.pkl')
groundnut_m=joblib.load('./groundnut.pkl')
cotton_m=joblib.load('./cotton.pkl')



@app.post('/top5res')
def top5res():
    # Get the request data
    try:
        inp_data = request.get_json()

        # Transform the data to match the model's input format
        X = np.array([[inp_data['dcode'], inp_data['year'], inp_data['scode']]])

        # Make the prediction
        prediction1 = wheat_m.predict(X)
        prediction2= sugarcane_m.predict(X)
        prediction3 = groundnut_m.predict(X)
        prediction4 = cotton_m.predict(X)
        prediction5 = soyabean_m.predict(X)

        
        # print(prediction1)
        # print(prediction2)
        # print(prediction3)
        # print(prediction4)
        # print(prediction5)
        # Return the prediction as JSON
        return jsonify({'predictions': [prediction1[0],prediction2[0],prediction3[0],prediction4[0],prediction5[0]] ,"success":True}),200


    except:
        return jsonify({'error': 'Unable to fetch at moment  '}), 500


if __name__ == '__main__':
    app.run(debug=True)



