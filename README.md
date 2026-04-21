# Fake News Detection

Detect whether a news claim is likely fake or true using an NLP + machine learning pipeline, with a Flask web interface for interactive predictions.

## Highlights

- End-to-end pipeline: data preparation, model training, and web deployment.
- Trained text classifier persisted with `joblib` as `model.pkl`.
- Flask UI for human-friendly predictions.
- JSON API support at `/predict` for programmatic use.
- Safety-oriented output mode (`Needs Verification`) for sensitive or uncertain claims.

## Project Structure

```text
Fake_news_Detection/
|-- data.csv
|-- retrain_model.py
|-- requirements.txt
|-- model.pkl                      # generated or expected in project root
|-- Fakenewsdetection.ipynb
|-- Data_set/news/news.csv
|-- Data-set/train.csv
|-- Data-set/test.csv
|-- Dataset/Fake.csv
|-- Dataset/True.csv
|-- politifact/politifact_fake.csv
|-- politifact/politifact_real.csv
`-- Model deployment using Flask/
          |-- app.py
          |-- test.txt
          `-- templates/index.html
```

## How It Works

1. Multiple public datasets are merged and normalized into two fields: `Article` and `label`.
2. Text is cleaned with regex-based preprocessing.
3. A scikit-learn pipeline is trained:
      1. `CountVectorizer`
      2. `TfidfTransformer`
      3. `LogisticRegression`
4. The trained model is saved as `model.pkl`.
5. Flask loads `model.pkl` and serves predictions through UI and API.

## Tech Stack

- Python
- Flask
- pandas
- scikit-learn
- numpy
- joblib

## Quick Start

### 1) Clone and enter the project

```bash
git clone https://github.com/your-username/Fake_news_Detection.git
cd Fake_news_Detection
```

### 2) Create and activate a virtual environment

Windows (Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the Flask app

From project root:

```bash
python "Model deployment using Flask/app.py"
```

Then open:

```text
http://127.0.0.1:5000/
```

## Retrain the Model

If you want to rebuild the classifier using the bundled datasets:

```bash
python retrain_model.py
```

This generates or overwrites `model.pkl` in the project root.

## API Usage

### Endpoint

- `POST /predict`

### JSON Request

```json
{
     "text": "Breaking: ..."
}
```

### Example cURL

```bash
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"Scientists confirm major discovery in climate research"}'
```

### Example Response (shape)

```json
{
     "prediction": 1,
     "label": "True News",
     "label_class": "true",
     "confidence": 92.14,
     "cleaned_text": "scientists confirm major discovery in climate research",
     "reason": null,
     "scores": {
          "fake": 7.86,
          "true": 92.14
     }
}
```

Possible labels:

- `Fake News`
- `True News`
- `Needs Verification`

## Deployment Recommendation

Recommended first deployment: Render.

Why:

- Simple Flask deployment workflow.
- Easy GitHub integration.
- Works well with the bundled `model.pkl` file.

### Render Setup

This repository already includes the Render pieces you need:

- `Procfile`
- `wsgi.py`
- `gunicorn` in `requirements.txt`

### Step-by-step deployment process

1. Push this project to GitHub.
2. Sign in to Render and click New + Web Service.
3. Connect the GitHub repository for this project.
4. Fill in the service details:
     1. Name: any name you want.
     2. Runtime: Python.
     3. Branch: your main branch.
5. Set the build command:

```bash
pip install -r requirements.txt
```

6. Set the start command:

```bash
gunicorn wsgi:app
```

7. Leave the port handling to Render. The app reads the `PORT` environment variable automatically.
8. Make sure `model.pkl` exists in the repository root before deploying.
9. Click Create Web Service and wait for the build to finish.
10. Open the generated Render URL and test the prediction form.

### Notes

- The Flask app is inside a folder with spaces, so Render should launch through `wsgi.py` instead of importing `app.py` directly.
- If you retrain the model later, commit the updated `model.pkl` before redeploying.

## Troubleshooting

- Error: `python app.py` from root fails.
     - Cause: `app.py` is inside `Model deployment using Flask/`.
     - Fix: run `python "Model deployment using Flask/app.py"`.

- Error: model file not found.
     - Ensure `model.pkl` exists in the project root.
     - If missing, run `python retrain_model.py`.

- Port already in use.
     - Stop other apps using `:5000`, then restart.

## Roadmap

- Add deep learning baseline (LSTM/Transformer) for comparison.
- Improve calibration and uncertainty estimation.
- Add automated tests for API and preprocessing.
- Add CI pipeline and Docker support.

## Disclaimer

This project is for educational and research purposes. Predictions are probabilistic and should not be used as the sole source of truth for critical decisions.

## Rights Reserved

All rights are reserved under ayush-py-c and aastha.
