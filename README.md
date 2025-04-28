# Bank Marketing Prediction Application

## Overview
This application is a machine learning-powered web service that predicts whether a customer will subscribe to a term deposit based on various demographic, financial, and campaign-related features. The application provides both a user-friendly web interface and API endpoints for making single and batch predictions.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API Endpoints](#api-endpoints)
- [Model Information](#model-information)
- [Deployment](#deployment)
- [Development](#development)

## Project Structure
```
bank_app/
├── app.py                # Main FastAPI application file
├── Dockerfile            # Docker configuration for containerization
├── fly_workflow.yml      # Fly.io deployment workflow configuration
├── fly.toml              # Fly.io configuration file
├── render.yml            # Render deployment configuration
├── requirements.txt      # Python dependencies
├── workflow.yml          # CI/CD workflow configuration
├── model/                # Directory containing model and inference code
│   ├── bank_model.joblib # Trained machine learning model
│   ├── inference.py      # Model inference functionality
│   ├── main.py           # Model training and evaluation script
│   ├── test.ipynb        # Jupyter notebook for model testing
│   └── data/             # Directory for model data
│       └── bank-full.csv # Banking dataset for model training
└── templates/            # HTML templates for the web interface
    └── index.html        # Main user interface template
```

## Features
- **Interactive Web Interface**: User-friendly form for submitting prediction requests
- **Single Prediction API**: Make predictions for individual customers
- **Batch Processing**: Upload CSV files for bulk predictions
- **Real-time Results**: Instant prediction results with probability scores
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Machine Learning**: scikit-learn, joblib
- **Data Processing**: pandas, numpy
- **Deployment**: Docker, Fly.io, Render

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Local Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd bank_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:8000`

### Docker Setup
1. Build the Docker image:
```bash
docker build -t bank-marketing-app .
```

2. Run the container:
```bash
docker run -p 8000:8000 bank-marketing-app
```

## Usage

### Web Interface
The web interface provides an intuitive form for entering customer information and getting predictions:

1. Navigate to `http://localhost:8000` in your web browser
2. Fill in the customer information form with the following details:
   - **Personal Information**: Age, job, marital status, education
   - **Financial Information**: Default status, housing loan, personal loan, account balance
   - **Contact Information**: Contact type, month, day, call duration
   - **Campaign Information**: Number of contacts, days since last contact, previous contacts, previous outcome
3. Click "Predict Subscription" to get results
4. View the prediction result (Yes/No) and probability scores

### Batch Processing
For processing multiple records at once:

1. Prepare a CSV file with customer data (following the same format as the example dataset)
2. In the web interface, scroll to the "Batch Processing" section
3. Upload your CSV file and click "Process Batch"
4. The system will process all records and return aggregated results

### API Endpoints

#### 1. Single Prediction
- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: application/json
- **Request Body Example**:
```json
{
  "job": "blue-collar",
  "marital": "married", 
  "education": "basic.4y",
  "default": "no", 
  "housing": "yes", 
  "loan": "no",
  "contact": "telephone", 
  "month": "may", 
  "poutcome": "nonexistent",
  "age": 40,
  "balance": 1500,
  "day": 15,
  "duration": 180,
  "campaign": 1,
  "pdays": 999,
  "previous": 0
}
```
- **Response Example**:
```json
{
  "prediction": "no",
  "probability": {
    "no": 0.85,
    "yes": 0.15
  }
}
```

#### 2. Batch Prediction
- **URL**: `/batch-predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Form Parameter**: `file` (CSV file)
- **Response Example**:
```json
{
  "batch_predictions": [
    {
      "input": {...},
      "prediction": {...}
    },
    ...
  ],
  "total_processed": 10
}
```

## Model Information
The application uses a machine learning model trained on the [Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) from the UCI Machine Learning Repository. 

- **Model Type**: Classification model (likely a Random Forest or Gradient Boosting algorithm)
- **Features**: 16 input features covering demographic, financial, and campaign-related data
- **Target Variable**: Whether the client subscribed to a term deposit (binary: "yes", "no")
- **Model File**: `model/bank_model.joblib`

### Key Features
1. **Demographics**:
   - Age: Age of the customer
   - Job: Type of job
   - Marital: Marital status
   - Education: Education level
   
2. **Financial**:
   - Default: Has credit in default
   - Housing: Has housing loan
   - Loan: Has personal loan
   - Balance: Account balance
   
3. **Contact**:
   - Contact: Communication type
   - Month: Last contact month
   - Day: Last contact day
   - Duration: Last contact duration in seconds
   
4. **Campaign**:
   - Campaign: Number of contacts during campaign
   - Pdays: Days since client was last contacted
   - Previous: Number of contacts before this campaign
   - Poutcome: Outcome of previous marketing campaign

## Deployment
The application comes with configuration files for multiple deployment platforms:

### Fly.io Deployment
1. Install Flyctl: `curl -L https://fly.io/install.sh | sh`
2. Authenticate: `flyctl auth login`
3. Deploy: `flyctl deploy`

### Docker-based Deployment
1. Build the Docker image: `docker build -t bank-app .`
2. Push to your container registry
3. Deploy using the registry image

## Development

### Adding New Features
1. Make sure to update the `inference.py` file if changes to the model input/output format are required
2. Update the `index.html` template when adding new UI elements
3. Add new API endpoints in `app.py` as needed

### Testing
- Run unit tests (if available)
- Test the API endpoints using tools like Postman or curl
- Validate the model with the provided test notebook

---

*Created on April 28, 2025*