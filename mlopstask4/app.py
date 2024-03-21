from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    
    # Perform prediction using the loaded model
    prediction = model.predict([data['features']])[0]
    
    # Create a response with the prediction
    response = {'prediction': prediction}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
