import os
import uvicorn
import pandas as pd
import pycaret.classification as pycl
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException


load_dotenv()

app = FastAPI()

class Model:
    def __init__(self, modelname, bucketname):
        self.model = pycl.load_model(modelname, platform = 'aws', authentication = { 'bucket' : bucketname })

    def predict(self, data):
        predictions = pycl.predict_model(self.model, data=data).Label.to_list()
        return predictions

model_et = Model("et_deployed", "mlopsassignment180100112")
model_rf = Model("rf_deployed", "mlopsassignment180100112")


def results(model, file):
    if file.filename.endswith(".csv"):

        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)

        df = pd.read_csv("../data/creditcard.csv")

        if len(data.columns.difference(df.columns).array) or len(df.columns.difference(data.columns).array):
            raise HTTPException(status_code=415, detail="Column names different from the ones expected.")
        else:
            return {
                "Labels": model.predict(data)
            }

    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


@app.post("/{model_id}/predict")
async def get_predictions(model_id: str, file: UploadFile = File(...)):
    if model_id == 'et':
        model = model_et
    elif mode_id == 'rf':
        model = model_rf
    else:
        raise HTTPException(status_code=404, detail="Page not found.")
    return results(model , file)


if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    print("AWS Credentials missing. Please set required environment variables.")
    exit(1)
