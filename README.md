# Stockout Predictor

## Overview
This project builds a stockout prediction model for pharmacy inventory. It ingests transactional stock data, engineers features, trains a Gradient Boosting model, and produces reorder recommendations through a simple Flask UI.

## Data Source
- Local hospital inventory data (name redacted due to NDA).
- Data is supplied as CSV files that mimic the original monthly Excel exports.

## Data Information
Required columns:
- `Item name`
- `Transaction Date`
- `Opening Stock`
- `QTY transacted`
- `Closing Stock`
- `Type` (Increment/Decrement/Adjust)
- `Sales value`

The data represents item-level stock movements over time. Each row is a transaction with quantities and resulting stock levels.

## Feature Engineering
The modeling pipeline creates features such as:
- Rolling average of quantity (`rolling_qty_5`)
- Stock movement direction (`is_decrement`)
- Time since last transaction
- Calendar features (month, day of week)

## Models
Evaluated models:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting (selected)


**Chosen model:** Gradient Boosting  
**Threshold:** 0.10

## Recall Score and Why It Matters
From the trained model metadata:
- **Recall:** 0.9997
- **Precision:** 0.8030
- **F1:** 0.8906

We prioritize **recall** because the business goal is to **minimize stockouts**. Missing a true stockout (false negative) is more costly than flagging extra items for review.

## How It Works (High-Level)
1. **Ingest**: Validate and load CSV data.
2. **Prepare**: Clean data, engineer features, and build latest records per item.
3. **Predict**: Score stockout risk using Gradient Boosting and apply the 0.10 threshold.
4. **Recommend**: Return a list of items to review/reorder.

## Project Structure
```
ingest.py      # CSV loading and validation
prepare.py     # Cleaning and feature engineering
train.py       # Model training and artifact saving
predict.py     # Inference and recommendations
app.py         # Flask UI for upload + results
artifacts/     # Saved model + metadata
templates/     # HTML template
static/        # CSS styling
```

## Running the Project
**Train the model:**
```bash
python train.py --csv "Data/modeling_dataset.csv" --horizon-days 60 --stockout-level 10
```

**Start the UI (Flask):**
```bash
python app.py
```
Then open `http://127.0.0.1:5000`.

**Start the UI (Streamlit):**
```bash
streamlit run streamlit_app.py
```

## Deployment (Streamlit Community Cloud)
1. Push the repo to GitHub.
2. Ensure `requirements.txt` is present.
3. Add model artifacts by running training locally and committing the `artifacts/` folder,
   or run training after deployment if your data is available on the server.
4. In Streamlit Cloud, select the repo and set the app file to `streamlit_app.py`.

## Conclusion
This system provides a practical, explainable way to reduce pharmacy stockouts by flagging high-risk items early. The pipeline is modular and can be extended with richer features, new models, and automated retraining.
