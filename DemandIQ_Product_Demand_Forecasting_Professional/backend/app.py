
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE=Path(__file__).resolve().parent
MODELS=BASE/"models"
DATA=BASE.parent/"data"/"demand_dataset.xlsx"
app=FastAPI(title="DemandIQ API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

class Prediction(BaseModel):
    model: Literal["ridge","lasso"]="ridge"
    Year:int
    Month:int
    Product:str
    Price:float
    Discount_Percent:float
    Advertising_Spend:float
    Previous_Sales_Units:float
    Season:str
    Holiday:int
    Competitor_Price:float
    Stock_Available:float
    Promotion:int

class RetrainRequest(BaseModel):
    ridge_alpha:float=1.0
    lasso_alpha:float=0.1
    lasso_max_iter:int=10000

def build_pipeline(model_name:str, ridge_alpha:float=1.0, lasso_alpha:float=0.1, lasso_max_iter:int=10000):
    df=pd.read_excel(DATA).drop_duplicates()
    X=df.drop(columns=["Demand_Units"])
    y=df["Demand_Units"]
    numeric=X.select_dtypes(exclude="object").columns.tolist()
    categorical=X.select_dtypes(include="object").columns.tolist()
    prep=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    estimator=Ridge(alpha=ridge_alpha) if model_name=="ridge" else Lasso(alpha=lasso_alpha,max_iter=lasso_max_iter)
    return df,X,y,Pipeline([("prep",prep),("model",estimator)])

def evaluate_models(ridge_alpha:float=1.0, lasso_alpha:float=0.1, lasso_max_iter:int=10000):
    df,X,y,_=build_pipeline("ridge",ridge_alpha,lasso_alpha,lasso_max_iter)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    output={"timestamp":datetime.now(timezone.utc).isoformat(),"dataset":{"total_records":len(df),"input_features":len(X.columns),"target":"Demand_Units","training_records":len(X_train),"testing_records":len(X_test),"split_ratio":"80 / 20","duplicates_removed":int(pd.read_excel(DATA).shape[0]-len(df))},"preprocessing":{"numeric_scaling":"StandardScaler","categorical_encoding":"OneHotEncoder"},"models":{}}
    for name in ("ridge","lasso"):
        _,_,_,pipe=build_pipeline(name,ridge_alpha,lasso_alpha,lasso_max_iter)
        pipe.fit(X_train,y_train)
        predictions=pipe.predict(X_test)
        errors=(y_test.to_numpy()-predictions).tolist()
        transformed_names=pipe.named_steps["prep"].get_feature_names_out().tolist()
        coefficients=pipe.named_steps["model"].coef_.tolist()
        coefficient_rows=[{"feature":feature,"coefficient":float(value)} for feature,value in zip(transformed_names,coefficients)]
        input_features={column:[] for column in X.columns}
        for row in coefficient_rows:
            transformed=row["feature"].replace("num__","").replace("cat__","")
            source=next((column for column in X.columns if transformed==column or transformed.startswith(f"{column}_")),transformed)
            input_features.setdefault(source,[]).append(row["coefficient"])
        selected_input_features=[column for column,values in input_features.items() if any(abs(value)>1e-8 for value in values)]
        output["models"][name]={
            "status":"Ready","alpha":ridge_alpha if name=="ridge" else lasso_alpha,"regularization":"L2" if name=="ridge" else "L1","max_iterations":None if name=="ridge" else lasso_max_iter,
            "metrics":{"mae":float(mean_absolute_error(y_test,predictions)),"mse":float(mean_squared_error(y_test,predictions)),"rmse":float(np.sqrt(mean_squared_error(y_test,predictions))),"r2":float(r2_score(y_test,predictions))},
            "actual_vs_predicted":[{"actual":float(actual),"predicted":float(predicted)} for actual,predicted in zip(y_test,predictions)],
            "residuals":[{"predicted":float(predicted),"residual":float(error)} for predicted,error in zip(predictions,errors)],
            "errors":errors,"mean_error":float(np.mean(errors)),"median_error":float(np.median(errors)),"max_error":float(np.max(errors)),"min_error":float(np.min(errors)),
            "coefficients":coefficient_rows,"selected_coefficients":sum(abs(value)>1e-8 for value in coefficients),"zero_coefficients":sum(abs(value)<=1e-8 for value in coefficients),"transformed_features":len(coefficients),"selected_input_features":selected_input_features,"zero_input_features":[column for column in X.columns if column not in selected_input_features]
        }
        joblib.dump(pipe,MODELS/f"{name}.joblib")
    output["models"]["ridge"]["features_used"]=len(X.columns)
    output["models"]["lasso"]["features_used"]=len(X.columns)
    return output

ML_RESULTS=evaluate_models()

@app.get("/")
def root(): return {"name":"DemandIQ API","status":"running"}

@app.post("/predict")
def predict(p:Prediction):
    model=joblib.load(MODELS/f"{p.model}.joblib")
    data=p.model_dump(exclude={"model"})
    value=float(model.predict(pd.DataFrame([data]))[0])
    return {"model":p.model,"prediction":max(0,value)}

@app.get("/ml-results")
def ml_results():
    return ML_RESULTS

@app.post("/retrain")
def retrain(request:RetrainRequest):
    global ML_RESULTS
    ML_RESULTS=evaluate_models(request.ridge_alpha,request.lasso_alpha,request.lasso_max_iter)
    return ML_RESULTS

@app.get("/health")
def health(): return {"status":"healthy"}
