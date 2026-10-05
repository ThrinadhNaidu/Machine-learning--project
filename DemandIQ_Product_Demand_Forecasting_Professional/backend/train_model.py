
from pathlib import Path
import pandas as pd, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"demand_dataset.xlsx"
OUT=Path(__file__).resolve()/"models"
df=pd.read_excel(DATA).drop_duplicates()
X=df.drop(columns=["Demand_Units"]); y=df["Demand_Units"]
num=X.select_dtypes(exclude="object").columns.tolist(); cat=X.select_dtypes(include="object").columns.tolist()
prep=ColumnTransformer([
("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
for name,est in [("ridge",Ridge(alpha=1.0)),("lasso",Lasso(alpha=.1,max_iter=10000))]:
    pipe=Pipeline([("prep",prep),("model",est)]);pipe.fit(Xt,yt)
    pred=pipe.predict(Xv)
    print(name,{"MAE":mean_absolute_error(yv,pred),"MSE":mean_squared_error(yv,pred),"RMSE":np.sqrt(mean_squared_error(yv,pred)),"R2":r2_score(yv,pred)})
    joblib.dump(pipe,OUT/f"{name}.joblib")
