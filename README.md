# AI Complaint Management System

A machine learning-based complaint management prototype designed to classify customer complaints and predict their urgency level. The system helps organizations logically organize, prioritize, and track complaints efficiently.

## Project Overview

This project is built as a prototype for an AI-powered complaint management system. It allows users to enter a complaint description, and the model predicts:

- the complaint category
- the complaint priority

The system is intended to help support or admin teams review complaints quickly, assign responsibilities, and monitor whether complaints are new, in progress, resolved, escalated, or closed.

## Problem Statement

Many businesses receive a large number of complaints daily. Manually sorting and prioritizing them can be slow and inconsistent. This project demonstrates how machine learning and natural language processing can help automate classification and urgency detection to improve service response and complaint handling.

## Features

- Complaint entry through a simple user-friendly interface
- Machine learning-based category prediction
- Priority detection for each complaint
- Dashboard overview of complaint trends
- Admin-only access to manage complaints
- Complaint status tracking:
  - New
  - In Progress
  - Resolved
  - Escalated
  - Closed
- Timeline tracking of actions taken on each complaint
- Secure prototype admin flow

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib

## ML Workflow

1. Complaint text is collected from the user
2. Text is transformed using TF-IDF
3. The category prediction model classifies the complaint into a category
4. The priority prediction model estimates urgency
5. The complaint is saved into the system for admin review
6. Admin updates the status and tracks the complaint lifecycle

## Project Structure

```text
AI_Complaint_System/
├── app/
│   └── app.py
├── dataset/
│   └── complaints.csv
├── models/
│   ├── category_model.pkl
│   ├── category_vectorizer.pkl
│   ├── priority_model.pkl
│   └── priority_vectorizer.pkl
├── src/
│   ├── data_analysis.py
│   ├── eda.py
│   ├── generate_dataset.py
│   ├── priority_model.py
│   └── train_model.py
├── requirements.txt
├── README.md
└── venv/
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd AI_Complaint_System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app/app.py
```

## Demo Admin Credentials

For the prototype admin panel:

- Username: `admin`
- Password: `Admin@123`

## Sample Use Case

A user enters a complaint such as:

> "I was charged twice for my order and I did not receive the product."

The AI model predicts:

- Category: Payment
- Priority: High

The admin can then review the complaint, assign it to the correct team, update the status, and mark it as resolved when completed.

## Project Status

This is a prototype machine learning application for demonstration and portfolio purposes. It is intended to showcase complaint classification, priority prediction, and admin workflow management.

## Future Enhancements

- real user login and authentication
- database integration for persistent complaint storage
- email notifications to users and admins
- role-based access for support staff and managers
- analytics and chart-based reporting
- export complaint reports to CSV/PDF
- production-grade security and deployment setup

## License

This project is intended for educational and portfolio use.

## Acknowledgements

This project demonstrates a practical ML application in complaint management and customer support operations.
