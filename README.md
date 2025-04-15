# AI-Health-Assistant
AI health Assistant Chatbot using my own data



# AI Health Assistant

This project is a Flask-based web application that provides several machine learning-powered features such as:

- Counseling Response Generation using a GPT-2 model.
- Medication Information Generation using a GPT-2 model.
- Diabetes Classification using a Random Forest classifier.
- Medicine Classification using a K-Nearest Neighbors (KNN) model.
- General Chat powered by LLaMA 3.1 API hosted on Groq Cloud for AI-powered conversations.

The project is divided into two main parts: Backend (Flask) and Frontend (HTML, CSS, JavaScript), with a connection to pre-trained machine learning models.
### Project Setup

-  **System Requirements:**
    - Python 3.8+
    - Flask
    - Transformers library (for GPT-2 models)
    - Joblib (for loading pre-trained models)
    - Langchain Groq (for LLaMA integration)
    - Frontend: HTML, CSS, JavaScript

- **API Details**
    - https://wandb.ai/rheddevil78-student
    
    ├── The Weights & Biases (W&B) platform is designed to help AI teams manage and accelerate their machine learning workflows. It provides tools for:
        ├── Tracking experiments: W&B allows you to track, visualize, and compare machine       learning experiments with easy-to-use dashboards.
        ├── Hyperparameter optimization: Automate the process of hyperparameter tuning through sweeps.
        ├── Model versioning and sharing: It offers model registries for managing versions and sharing models and datasets.
        ├── Pipeline management: You can manage machine learning pipelines and integrate them into the workflow.


- **Project Structure:**
    ```
    AI Health Assistant/
    │
    ├── backend/
    │   ├── models/
    │   │   ├── mental_health_model/
    │   │   ├── medication_info/
    │   │   ├── diabetes_model/
    │   │   ├── medication_classification_model/
    │   ├── utils.py
    ├── frontend/
    │   ├── index.html
    │   ├── styles.css
    │   ├── script.js
    ├── app.py
    ├── requirements.txt
    ├── AI health Assisstant.gif
    

### Backend

**Counseling Response Generation:**
- Generates counseling-related responses using a GPT-2 mental health model.

**Medication Information Generation:**
- Provides medication-related responses using a GPT-2 medication model.

**Diabetes Classification:**
- Classifies users as diabetic or non-diabetic based on glucose, BMI, and age using a Random Forest classifier.

**Medicine Classification:**
- Predicts suitable medications based on gender, blood type, medical condition, and test results using a K-Nearest Neighbors (KNN) model.

**General Chat:**
- Offers general chat responses using LLaMA 3.1 API hosted on Groq Cloud for AI-powered conversations.


### Frontend

**Diabetes Classification Tab:**
- Form input for glucose, BMI, and age to classify diabetes risk.

**Medicine Classification Tab:**
- Input fields for gender, blood type, medical condition, and test results to classify appropriate medications.

**Counseling and Medication Tabs:**
- Text inputs for receiving AI-generated responses for counseling and medication questions.

**General Chat Tab:**
- General-purpose chatbot powered by LLaMA 3.1 for natural conversations.

**Dark Mode:**
- Toggle dark mode for user interface customization.


### Usage

1. **Access the Application:** Users interact with the web interface, accessible through a browser once the Flask server is running.

2. **Input Data:** Users provide medical-related information or general queries depending on the feature they want to use.

3. **Receive Responses:** Based on the input, AI models provide responses such as classification results (diabetes, medicine) or generated text (counseling, medication, chat).

4. **Interactive Interface:** Users can toggle between different features, making it suitable for general chat, medical insights, or counseling help.


### WebApp
