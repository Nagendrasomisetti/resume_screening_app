# 📄 Resume Screening App

A simple web application built with Streamlit to automate the process of screening resumes using a machine learning model.

## Features

- Paste resume text and get instant screening results
- Uses a trained Logistic Regression model with TF-IDF vectorization
- User-friendly interface with clear feedback

## How It Works

1. Paste the text of a resume into the app.
2. Click **Screen Resume**.
3. The app predicts if the resume should be selected for the next round.

## Requirements

- Python 3.7+
- streamlit
- scikit-learn
- pandas
- joblib

## Getting Started

1. Clone this repository.
2. Place `model.pkl` and `vectorizer.pkl` in the project folder.
3. Install dependencies:
    ```
    pip install streamlit scikit-learn pandas joblib
    ```
4. Run the app:
    ```
    streamlit run resume_screening.py
    ```
5. Open the provided local URL in your browser.

6. ## Project Structure

```
resume_screening_app/
│
├── resume_screening.py
├── model.pkl
├── vectorizer.pkl
└── README.md
```

## License

This project is for educational purposes.
