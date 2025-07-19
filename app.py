from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
def load_model():
    with open('model/logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse inputs from request
        data = request.form.to_dict()

        # Convert inputs into DataFrame
        df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(df)
        feature_importance = model.coef_

        result = {
            'prediction': prediction[0],
            'feature_importance': feature_importance.tolist()
        }
        return jsonify(result)
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
