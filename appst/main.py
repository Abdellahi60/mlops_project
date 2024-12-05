from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import mlflow
import pandas as pd
from src.clean_data import clean_data,clean_data1
from fastapi import FastAPI, File, UploadFile,Form
import pandas as pd
import mlflow
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from preprocess import num_transformer
from preprocess import cat_transformer
from preprocess import categorical_features
from preprocess import numeric_features
import numpy as np
from fastapi.staticfiles import StaticFiles
# Initialisation de l'app FastAPI
app = FastAPI()

# Spécifier le répertoire où se trouvent les fichiers HTML
templates = Jinja2Templates(directory="templates")

# Configurer MLflow avec vos identifiants DagsHub
os.environ['MLFLOW_TRACKING_USERNAME'] = "Abdellahi60"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "6870def210d626bd8f3c95f27c52207fb377ce80"

# Configuration MLflow
mlflow.set_tracking_uri('https://dagshub.com/Abdellahi60/mlops-bootcamp.mlflow')
df_mlflow = mlflow.search_runs(filter_string="metrics.F1_score_test<1")
run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']
logged_model = f'runs:/{run_id}/ML_models'
model = mlflow.pyfunc.load_model(logged_model)

# Endpoint pour la page d'accueil avec formulaire de téléchargement de CSV
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Bienvenue dans l'application de prédiction des étudiants"})

# Endpoint pour effectuer des prédictions à partir d'un fichier CSV
@app.post("/predict/csv")
async def predict_csv(request: Request,file: UploadFile = File(...)):
    # Lire le fichier CSV
    data = pd.read_csv(file.file)

    # Appliquer le nettoyage des données
    preprocessed_data = clean_data(data)

    # Effectuer les prédictions
    predictions = model.predict(preprocessed_data)
    
    prediction_labels = {0: "admis", 1: "ajournée", 2: "sessionnaire"}
    data["Prediction"] = [prediction_labels[pred] for pred in predictions]
    data_dict = data.to_dict(orient="records")
    # Retourner les prédictions
    return templates.TemplateResponse("result.html",{"request": request,"data_dict":data_dict} )

@app.post('/prediction_result')
async def prediction_result(
    request: Request,
    nom: str = Form(...),
    etablissement: str = Form(...),
    serie: str = Form(...),
    centre: str = Form(...),
    willaya: str = Form(...),
    moughataa: str = Form(...),
    age: int = Form(...)
):
        # Example instance for prediction
        example_instance = pd.DataFrame([{
            'Etablissement': etablissement,
            'Serie,x': serie,
            'Centre': centre,
            'Willaya': willaya,
            'moughataa': moughataa,
            'Age': age
        }])
        preprocessed_data = clean_data1(example_instance)

    # Effectuer les prédictions
        predictions = model.predict(preprocessed_data)
        example_instance['name']=nom
        example_instance['Prediction'] = predictions
        example_instance['Prediction'] = example_instance['Prediction'].replace({
            0: "Admis",
            1: "Ajournée",
            2: "Sessionnaire"
        })
        result=example_instance.to_dict(orient="records")
        return templates.TemplateResponse("prediction_result.html",{"request": request,"result":result} )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)