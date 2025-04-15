function showTab(tabId) {
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => {
        if (tab.id === tabId) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
}

function submitCounseling() {
    const question = document.getElementById('counseling-question').value;
    fetch('/api/counseling', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('counseling-response').innerText = data.response;
    })
    .catch(error => console.error('Error:', error));
}

function submitMedication() {
    const question = document.getElementById('medication-question').value;
    fetch('/api/medication', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('medication-response').innerText = data.response;
    })
    .catch(error => console.error('Error:', error));
}

function submitDiabetes() {
    const glucose = document.getElementById('glucose').value;
    const bmi = document.getElementById('bmi').value;
    const age = document.getElementById('age-diabetes').value;
    fetch('/api/diabetes_classification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ glucose, bmi, age })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('diabetes-response').innerText = data.result;
    })
    .catch(error => console.error('Error:', error));
}

function submitMedicine() {
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const bloodType = document.getElementById('blood-type').value;
    const medicalCondition = document.getElementById('medical-condition').value;
    const testResults = document.getElementById('test-results').value;
    
    fetch('/api/medicine_classification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ age, gender, blood_type: bloodType, medical_condition: medicalCondition, test_results: testResults })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('medicine-response').innerText = data.medicine;
    })
    .catch(error => console.error('Error:', error));
}

function submitGeneral() {
    const question = document.getElementById('general-question').value;
    fetch('/api/general', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
    .then(response => response.json())
    .then(data => {
        // document.getElementById('general-response').innerText = data.response;
        document.getElementById('general-response').innerText = "Please Try With Another Question";
    })
    .catch(error => console.error('Error:', error));
}
