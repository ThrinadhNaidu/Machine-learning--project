# DemandIQ — Product Demand Forecasting

A professional ML analytics website for **Product Demand Forecasting Using Ridge and Lasso Regression**.

## Included
- Professional multi-page analytics dashboard
- Real 504-row / 13-column project dataset
- Interactive Chart.js visualizations
- Dataset search, filtering and export
- Excel upload using SheetJS
- CSV / Excel / PDF export
- Ridge + Lasso evaluation metrics
- FastAPI prediction backend with trained scikit-learn pipelines
- Responsive desktop/tablet/mobile design

## Project flow
Excel data → cleaning → preprocessing → Ridge/Lasso → evaluation → analytics → prediction → export.

## Run the website
Open `frontend/` in VS Code and use **Live Server**. Start with `frontend/index.html`.

## Run the real prediction API
Open a terminal in `backend/`:

```bash
pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload --port 8000
```

Then open the website through Live Server and use **Demand Prediction**.

The frontend first tries `http://127.0.0.1:8000/predict`. If the backend is not running, it shows a clearly labeled demo estimate instead of pretending it is a real ML prediction.

## Dataset
The included Excel file is:
`data/demand_dataset.xlsx`

The original dataset contains 504 records and 13 columns. Data-quality inspection found 4 duplicate rows and missing values in several numeric input columns. The model training script removes exact duplicates and imputes numeric/categorical missing values inside the preprocessing pipeline.

## Presentation order
1. Dashboard — project overview
2. Dataset Explorer — dataset and data quality
3. Analytics — business relationships and charts
4. ML Models — Ridge vs Lasso and metrics
5. Demand Prediction — live model demonstration
6. Reports & Export — final outputs

## Technologies
HTML5, CSS3, JavaScript, Chart.js, SheetJS, jsPDF, Python, FastAPI, pandas, scikit-learn, joblib.
