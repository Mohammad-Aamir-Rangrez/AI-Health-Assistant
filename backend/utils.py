# backend/utils.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_groq import ChatGroq
import torch
import requests
import joblib
import pandas as pd

# Load the trained model and tokenizer : Counselling
counseling_model = GPT2LMHeadModel.from_pretrained('backend\\models\\mental_health_model')
counselling_tokenizer = GPT2Tokenizer.from_pretrained('backend\\models\\Mental_health_model')

# Load the trained model and tokenizer : Medication
medication_model = GPT2LMHeadModel.from_pretrained('backend\\models\\medication_info')
medication_tokenizer = GPT2Tokenizer.from_pretrained('backend\\models\\medication_info')

# Load the trained Random Forest model and StandardScaler
diabetes_model = joblib.load('backend\\models\\diabetes\\diabetes_random_forest.joblib')
diabetes_scaler = joblib.load('backend\\models\\diabetes\\standard_scaler.joblib')

# Load the model, encoders, and scaler
knn = joblib.load('backend\\models\\Madicine_classification\\knn_model.pkl')
label_encoders = joblib.load('backend\\models\\Madicine_classification\\label_encoders.pkl')
age_scaler = joblib.load('backend\\models\\Madicine_classification\\age_scaler.pkl')
medication_encoder = joblib.load('backend\\models\\Madicine_classification\\medication_encoder.pkl')

# Load the trained model and tokenizer : General Chat
General_Chat_model = GPT2LMHeadModel.from_pretrained('backend\\models\\General_chat')
General_Chat_tokenizer = GPT2Tokenizer.from_pretrained('backend\\models\\General_chat')



# Diabetes Classifier
def classify_diabetes(glucose, bmi, age):
    # Normalize the input features
    input_features = [[glucose, bmi, age]]
    input_features_norm = diabetes_scaler.transform(input_features)

    # Make predictions
    prediction = diabetes_model.predict(input_features_norm)[0]
    prediction_probability = diabetes_model.predict_proba(input_features_norm)[0] * 100

    diabetic_probability = prediction_probability[prediction].item()

    if prediction == 0:
        result = "Non Diabetic"
    else:
        result = "Diabetic"

    # Format the output as: "Non Diabetic | 72%"
    formatted_result = f"{result} | {diabetic_probability:.1f}%"
    return formatted_result


# Medicine Classifier
def classify_medicine(new_data):
    # Convert dictionary to DataFrame
    new_data_df = pd.DataFrame(new_data)
    
    # Encode the new data using the saved label encoders
    for column in ['Gender', 'Blood Type', 'Medical Condition', 'Test Results']:
        new_data_df[column] = label_encoders[column].transform(new_data_df[column])

    # Normalize the 'Age' column in the new data
    new_data_df['Age'] = age_scaler.transform(new_data_df[['Age']])

    # Make predictions
    predictions = knn.predict(new_data_df)

    # Decode the predictions back to the original medication names
    predicted_medications = medication_encoder.inverse_transform(predictions)

    return predicted_medications


# Generate Counseling Response
def generate_counseling_response(prompt):
    inputs = counselling_tokenizer.encode(prompt, return_tensors="pt")
    outputs = counseling_model.generate(inputs, max_length=150, num_return_sequences=1,temperature=0.8,top_k=50,top_p=0.9,no_repeat_ngram_size=2, pad_token_id=counselling_tokenizer.eos_token_id)

    # Decode the generated output
    response = counselling_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()  # Remove the prompt from the response

    return response


# Generate Medication Response
def generate_medication_response(prompt):
    inputs = medication_tokenizer.encode(prompt, return_tensors="pt")
    outputs = medication_model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.8,top_k=50,top_p=0.9,no_repeat_ngram_size=2,pad_token_id=medication_tokenizer.eos_token_id)

    # Decode the generated output
    response = medication_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()  # Remove the prompt from the response

    return response


# Llama 3.1 Integration as a General Tab
# llm = ChatGroq(
#     temperature=0,
#     groq_api_key='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',
#     model_name="llama-3.1-70b-versatile"
# )

def format_response(response):
    # Add line breaks and make it easier to read
    response = response.replace("**", "").replace("*", "").replace("  ", "\n").strip()
    lines = response.split("\n")
    formatted_response = ""
    for line in lines:
        formatted_response += f"<p>{line.strip()}</p>"
    return formatted_response


def generate_general_chat_response(prompt): #  use_llama=False
    # if use_llama:
    #     try:
    #         response = llm.invoke(prompt, timeout=10)
    #         formatted_response = format_response(response.content)
    #         return formatted_response
    #     except requests.exceptions.Timeout:
    #         return "Error: The request to LLaMA timed out. Please try again later."
    #     except Exception as e:
    #         return f"Error: {str(e)}"
    # else:
        inputs = General_Chat_tokenizer.encode(prompt, return_tensors="pt")
        outputs = General_Chat_model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=General_Chat_tokenizer.eos_token_id
        )
        response = General_Chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response

