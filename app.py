from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('model.h5', compile=False)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001)
)

def softmax(x):
    z = np.exp(x)
    s = z / z.sum(axis=0)
    return s

def extract_features(data):
    feature = []
    # ml_features
    ml_features = data['mlFeatures']
    feature_1 = [
        ml_features['AIR_INTAKE_TEMP'], ml_features['ENGINE_COOLANT_TEMP'], ml_features['ENGINE_LOAD'], ml_features['ENGINE_POWER'],
        ml_features['ENGINE_RPM'], ml_features['INTAKE_MANIFOLD_PRESSURE'], ml_features['SPEED'], ml_features['THROTTLE_POS'],
        ml_features['TIMING_ADVANCE']
    ]
    feature.extend(feature_1)
    
    # time
    time_data = data['timeData']
    feature_2 = [time_data['DAYS'], time_data['HOURS'], time_data['MONTHS']]
    feature.extend(feature_2)

    return feature

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if 'input' not in data:
            return jsonify({'error': 'Invalid input data'}), 400

        input_data = data['input']
        if not input_data:
            return jsonify({'error': 'Empty input data'}), 400

        # Log the received input data for debugging
        print(f"Received input data: {input_data}")

        input_np = np.array(extract_features(input_data)).reshape(-1, 12)
        print(input_np)

        prediction = np.argmax(model.predict(input_np), axis=1).tolist()
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Log the exception for debugging
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
