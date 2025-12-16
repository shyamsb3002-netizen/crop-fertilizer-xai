from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

# Load scaler and models
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

# Load models
with open('fertilizer_model_1.pkl', 'rb') as file:
    fertilizer_model = pickle.load(file)

with open('crop_model.pkl', 'rb') as file:
    crop_model = pickle.load(file)

# Load crop dataset
df = pd.read_csv('Crop_recommendation.csv')
crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
feature_names_map = {
    'N': 'Nitrogen',
    'P': 'Phosphorus',
    'K': 'Potassium',
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'ph': 'pH',
    'rainfall': 'Rainfall'
}

# Initialize LIME explainer for crop model
crop_explainer = lime.lime_tabular.LimeTabularExplainer(
    df[crop_features].values,
    feature_names=crop_features,
    class_names=np.unique(df['label']),
    discretize_continuous=True
)

# Load fertilizer dataset
df_fertilizer = pd.read_csv('f2.csv')  # Ensure correct filename
fertilizer_features = ['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Phosphorous', 'Potassium']
fertilizer_feature_map = {
    'Temparature': 'Temperature',
    'Humidity': 'Humidity',
    'Moisture': 'Soil Moisture',
    'Soil_Type': 'Soil Type',
    'Crop_Type': 'Crop Type',
    'Nitrogen': 'Nitrogen Level',
    'Phosphorous': 'Phosphorus Level',
    'Potassium': 'Potassium Level'
}

# Convert categorical features to numerical encoding
df_fertilizer['Soil_Type'], _ = pd.factorize(df_fertilizer['Soil_Type'])
df_fertilizer['Crop_Type'], _ = pd.factorize(df_fertilizer['Crop_Type'])

# Ensure all values are numeric
df_fertilizer[fertilizer_features] = df_fertilizer[fertilizer_features].apply(pd.to_numeric, errors='coerce')

# Initialize LIME explainer for fertilizer model
fertilizer_explainer = lime.lime_tabular.LimeTabularExplainer(
    df_fertilizer[fertilizer_features].values.astype(float),  # Convert to float
    feature_names=fertilizer_features,
    class_names=[str(cls) for cls in np.unique(df_fertilizer['Fertilizer'])],
    discretize_continuous=True
)

# Fertilizer name mapping
fertilizer_mapping = {
    1: 'Urea', 2: 'TSP', 3: 'Superphosphate', 4: 'Potassium sulfate',
    5: 'Potassium chloride', 6: 'DAP', 7: '28-28', 8: '20-20', 9: '17-17-17',
    10: '15-15-15', 11: '14-35-14', 12: '14-14-14', 13: '10-26-26', 14: '10-10-10'
}

# Temperature Scaling Function for Confidence Calibration
def temperature_scaled_probs(probs, T=1.5):
    exp_probs = np.exp(probs / T)
    return exp_probs / np.sum(exp_probs)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop')
def option1():
    return render_template('crop.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/crop/predict', methods=['POST'])
def predict_crop():
    n = float(request.form['nitrogen'])
    p = float(request.form['phosphorus'])
    k = float(request.form['potassium'])
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Prepare input features for prediction
    X = np.array([[n, p, k, temp, humidity, ph, rainfall]])

    # Get prediction and predicted probabilities
    prediction = crop_model.predict(X)[0]
    prediction_probabilities = crop_model.predict_proba(X)[0]

    # Calculate the confidence score (probability of the predicted class)
    confidence_score = prediction_probabilities[np.argmax(prediction_probabilities)] * 100

    # Generate LIME explanation for this prediction
    exp = crop_explainer.explain_instance(X[0], crop_model.predict_proba, num_features=len(crop_features))
    
    # Convert explanation to a readable format
    explanation = exp.as_list()

    # Prepare the explanation sentence
    explanation_sentence = "The recommended crop is primarily influenced by: "
    for feature, importance in explanation[:3]:  # Top 3 most important features
        feature_key = feature.split()[0]
        feature_name = feature_names_map.get(feature_key, feature_key)
        explanation_sentence += f"{feature_name} (Importance: {importance:.4f}), "
    explanation_sentence = explanation_sentence.rstrip(", ") + "."

    # Generate feature importance graph
    feature_importances = [importance for _, importance in explanation]
    feature_names = [feature for feature, _ in explanation]
    
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance for Crop Recommendation')

    # Save graph to a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    img.close()

    # Return the prediction, confidence score, explanation, and graph
    return render_template('crop_predict.html', 
                           prediction_text=f'Predicted Crop: {prediction}',
                           confidence_score=f'Confidence: {confidence_score:.2f}%',
                           explanation_sentence=explanation_sentence,
                           img_data=img_base64)

    # Return the prediction, confidence score, explanation, and graph
    return render_template('crop_predict.html', 
                           prediction_text=f'Predicted Crop: {prediction}',
                           confidence_score=f'Confidence: {confidence_score:.2f}%',
                           explanation_sentence=explanation_sentence,
                           img_data=img_base64)

@app.route('/fertilizer/predict', methods=['POST'])
def predict_fertilizer():
    soiltype = float(request.form['soil'])
    n = float(request.form['nitrogen'])
    p = float(request.form['phosphorus'])
    k = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    crop_name = float(request.form['croptype'])
    moisture = float(request.form['Moisture'])

    X = np.array([[temperature, humidity, moisture, soiltype, crop_name, n, p, k]])
    standardized_input = scaler.transform(X)

    prediction = fertilizer_model.predict(standardized_input)[0]
    prediction_name = fertilizer_mapping.get(int(prediction), f"Unknown Fertilizer ({int(prediction)})")
    prediction_probabilities = fertilizer_model.predict_proba(standardized_input)[0]

    # Apply Temperature Scaling for Confidence Calibration
    prediction_probabilities = temperature_scaled_probs(prediction_probabilities)
    confidence_score = (np.max(prediction_probabilities) / np.sum(prediction_probabilities)) * 1000

    # Generate LIME explanation
    exp = fertilizer_explainer.explain_instance(
        standardized_input[0], 
        fertilizer_model.predict_proba, 
        num_features=len(fertilizer_features)
    )
    explanation = exp.as_list()

    # Ensure only proper feature names appear
    cleaned_explanation = []
    for feature, importance in explanation:
        feature_key = feature.split()[0]  # Extract feature name
        feature_name = fertilizer_feature_map.get(feature_key, feature_key)  # Map to readable name
        cleaned_explanation.append(f"{feature_name} (Impact: {importance:.4f})")

    # Construct user-friendly explanation sentence
    # Construct the explanation sentence
    explanation_sentence = "The recommended fertilizer is influenced by "
    explanation_sentence += ", ".join(cleaned_explanation[:3]) + "."


    # Generate feature importance plot
    feature_importances = [importance for _, importance in explanation]
    feature_names = [fertilizer_feature_map.get(feature.split()[0], feature.split()[0]) for feature, _ in explanation]

    plt.figure(figsize=(6, 4))
    plt.barh(feature_names, feature_importances, color='salmon')
    plt.xlabel('Importance')
    plt.title('Feature Importance for Fertilizer Recommendation')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    img.close()

    return render_template('fertilizer_predict.html',
                       prediction_text=f'Predicted Fertilizer: {prediction_name}',
                       confidence_score=f'Confidence: {confidence_score:.2f}%',
                       explanation_sentence=explanation_sentence,  # ✅ This will render correctly in HTML
                       img_data=img_base64)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



