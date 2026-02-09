# BERT-AI

A MERN stack application that provides an AI-powered chatbot interface with automatic ticket escalation when the chatbot cannot answer user questions.

---

## Step-by-Step Setup

### 1. Create and Activate a Virtual Environment

Create the virtual environment:

- python -m venv venv

Activate the virtual environment (Windows CMD):

- venv\Scripts\activate

Activate the virtual environment (Git Bash / PowerShell):

- source venv/Scripts/activate

---

### 2. Install Dependencies

Install all required packages:

- python -m pip install -r requirements.txt

---

### 3. Fix Dependency Issues (Optional)

If you encounter installation errors, try upgrading pip and reinstalling:

- python -m pip install --upgrade pip
- python -m pip install --upgrade -r requirements.txt

If you get an error related to `accelerate`, install it manually:

- python -m pip install accelerate>=1.1.0
