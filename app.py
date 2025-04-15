import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
from flask import Flask, jsonify, request, send_from_directory
from backend.utils import (
    generate_counseling_response,
    generate_medication_response,
    classify_diabetes,
    classify_medicine,
    generate_general_chat_response
)
import os

app = Flask(__name__, static_folder='frontend', template_folder='frontend')

# Serve the main HTML file for the frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# Serve the CSS files
@app.route('/styles.css')
def styles():
    return send_from_directory(app.static_folder, 'styles.css')


# Serve the JavaScript files
@app.route('/script.js')
def script():
    return send_from_directory(app.static_folder, 'script.js')


# Route for Counseling Model
@app.route('/api/counseling', methods=['POST'])
def counseling():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question is required."}), 400

    response = generate_counseling_response(question)
    return jsonify({"response": response})


# Route for Medication Info Model
@app.route('/api/medication', methods=['POST'])
def medication():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question is required."}), 400

    response = generate_medication_response(question)
    return jsonify({"response": response})


# Route for Diabetes Classification
@app.route('/api/diabetes_classification', methods=['POST'])
def diabetes_classification():
    data = request.json

    # Extract input features
    glucose = data.get('glucose')
    bmi = data.get('bmi')
    age = data.get('age')

    # Validate input data
    if glucose is None or bmi is None or age is None:
        return jsonify({"error": "Please provide glucose, bmi, and age."}), 400

    result = classify_diabetes(glucose, bmi, age)
    return jsonify({"result": result})


# Route for Medicine Classification
@app.route('/api/medicine_classification', methods=['POST'])
def medicine_classification():
    data = request.json

    # Extract input features
    age = data.get('age')
    gender = data.get('gender')
    blood_type = data.get('blood_type')
    medical_condition = data.get('medical_condition')
    test_results = data.get('test_results')

    # Validate input data
    if not (age and gender and blood_type and medical_condition and test_results):
        return jsonify({"error": "Please provide Age, Gender, Blood Type, Medical Condition, and Test Results."}), 400

    # Prepare the new data as a DataFrame
    new_data = {
        'Age': [int(age)],
        'Gender': [gender],
        'Blood Type': [blood_type],
        'Medical Condition': [medical_condition],
        'Test Results': [test_results]
    }

    # Call the classification function
    medicine = classify_medicine(new_data)
    return jsonify({"medicine": medicine[0]})

# Route for General Chat (Llama 3.1 API using Groq Cloud)
@app.route('/api/general', methods=['POST'])
def general_chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        # LLaMA-3.1 model response (hosted via Groq)
        llama_response = generate_general_chat_response(question, use_llama=True)

        # Fallback local model response
        local_response = generate_general_chat_response(question, use_llama=False)

        if llama_response.lower().startswith("error"):
            return jsonify({"error": llama_response}), 500

        # Return both responses (for testing/display/choice)
        return jsonify({
            "llama_response": llama_response,
            "local_response": local_response
        })

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)