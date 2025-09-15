# Engine Fault Predictor (OBD-II)

A backend service that trains and serves a TensorFlow/Keras model to predict future engine faults from historical OBD-II telemetry. The model learns patterns in sensor/time features and returns one of the predefined fault codes (including `NoError`).

---

## Features

* Trainable TensorFlow/Keras classification model for multi-class fault prediction.
* REST API for model inference (JSON input -> predicted fault code).
* Data preprocessing utilities (scaling, time-feature engineering, label encoding).
* Scripts to train, evaluate, save, and load models.

---

## Input & Output

### Input features (JSON keys expected by the API / training pipeline)

* `engine_power` (float)
* `engine_coolant_temp` (float)
* `engine_load` (float)  (Engine load value)
* `intake_manifold_pressure` (float)
* `engine_rpm` (float)
* `air_intake_temp` (float)
* `speed_throttle_position` (float)
* `timing_advance` (float)
* `hours` (int) — hour of day or engine-use hours; used as a numeric/time feature
* `months` (int)
* `day_of_week` (int) — 0 (Monday) .. 6 (Sunday)

> The backend expects a single JSON object or a list/array of objects for batch inference.

### Output

* `predicted_label` (string) — one of the target classes:

  * `NoError`
  * `P0133`
  * `C0300`
  * `P0079/P3000/P2004`
  * `P0078/U1004/P3000`
  * `P0079/C1004/P3000`
  * `P007E/P2036/P18FO`
  * `P007E/P2036/P18D0`
  * `P007F/P2036/P18D0`
  * `P0079/P1004/P3000`
  * `P007E/P2036/P18E0`
  * `P007F/P2036/P18E0`
  * `P0078/B0004/P3000`
  * `P007F/P2036/P18F0`
* `probabilities` (object) — mapping from class -> predicted probability
* `confidence` (float) — probability of the top-predicted class

---

## Example API usage

**Request**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "engine_power": 45.0,
    "engine_coolant_temp": 78.0,
    "engine_load": 33.2,
    "intake_manifold_pressure": 12.1,
    "engine_rpm": 2100,
    "air_intake_temp": 25.0,
    "speed_throttle_position": 18.0,
    "timing_advance": 5.2,
    "hours": 14,
    "months": 6,
    "day_of_week": 2
  }'
```

**Response**

```json
{
  "predicted_label": "P0079/C1004/P3000"
}
```

---

## Model architecture (suggested)

A straightforward starting architecture:

* Input -> shape = (12 , )
* hidden layer 01 -> Dense(25, relu)
* hidden layer 02 -> Dense(18, relu)
* Output Dense(14, activation=`softmax`)

Use Adam optimizer, learning-rate scheduling, and class weights (or oversampling) since faults are often imbalanced.

---

## Preprocessing notes

* Fill or drop missing sensor values (imputation suggested: forward-fill or median).
* Scale continuous features using `StandardScaler` or `RobustScaler` (save scaler to disk).
* Encode `day_of_week` and `month` using either one-hot or cyclic features (sin, cos) for continuity.
* Optionally create rolling-window features (e.g. moving averages) if model should use short-term history.

---

## Evaluation & Metrics

* Use confusion matrix, precision, recall, F1-score per class (fault codes often require high recall on critical faults).
* Use ROC-AUC per class (one-vs-rest) for class separability.
* Keep a hold-out test set with data from different vehicles/time periods if possible.

---

## Example dependencies (put in `requirements.txt`)

```
tensorflow>=2.10
keras
numpy
pandas
scikit-learn
flask
joblib
uvicorn
```
