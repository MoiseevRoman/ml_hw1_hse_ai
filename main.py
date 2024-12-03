
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd
import pickle
import re
import io

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]
    
def refact(text: str):
    pattern = r'[0-9.]+'
    match = re.search(pattern, text)
    if match:
        return float(match.group())
    else:
        return np.nan

def refact_torque(text: str) -> pd.Series:
    pattern = r"(\d+\.?\d*)\s*(Nm|nm|kgm)?\s*(?:@|at)?\s*(\d+(?:,\d+)?(?:-\d+(?:,\d+)?)?)\s*\(?\s*(Nm|nm|kgm)?@?\s*rpm\)?"
    match = re.search(pattern, text)
    if match:
        value, unit1, rpm, unit2 = match.groups()
        unit = (unit1 or unit2).lower() if (unit1 or unit2) else None
        rpm = rpm.replace(',', '')
        if unit == 'kgm':
            value = float(value) * 9.81
        else:
            value = float(value)
        if '-' in rpm:
            # Беру максимльное число оборотов в минуту 
            rpm = float(rpm.split('-')[-1])
        else:
            rpm = float(rpm)
        return pd.Series({'torque': value, 'max_torque_rpm': rpm})
    return pd.Series({'torque': np.nan, 'max_torque_rpm': np.nan})

def transform_predict(item: Item) -> float:
    
    df = pd.DataFrame([item.dict()])
    
    df['mileage'] = df['mileage'].apply(lambda x: refact(str(x)))
    df['engine'] = df['engine'].apply(lambda x: refact(str(x)))
    df['max_power'] = df['max_power'].apply(lambda x: refact(str(x)))
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(lambda x: refact_torque(str(x)))
    
    df['engine'] = df['engine'].apply(int)
    df['seats'] = df['seats'].apply(int)

    df['name'] = df['name'].apply(lambda x: x.split()[0])

    with open("pipline.pickle", 'rb') as f:
        pipeline = pickle.load(f)

    pred = pipeline.predict(df);
    
    return pred

def transform_predict_csv(df: pd.DataFrame) -> List[float]:
    df['mileage'] = df['mileage'].apply(lambda x: refact(str(x)))
    df['engine'] = df['engine'].apply(lambda x: refact(str(x)))
    df['max_power'] = df['max_power'].apply(lambda x: refact(str(x)))
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(lambda x: refact_torque(str(x)))
    
    df['engine'] = df['engine'].apply(int)
    df['seats'] = df['seats'].apply(int)

    df['name'] = df['name'].apply(lambda x: x.split()[0])

    with open("pipline.pickle", 'rb') as f:
        pipeline = pickle.load(f)

    pred = pipeline.predict(df);
    
    return pred    

    
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    res = transform_predict(item)
    return res[0]
    
@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> List[float]:
    contents = file.file.read()
    buffer = io.BytesIO(contents)
    df = pd.read_csv(buffer, sep=';')
    buffer.close()
    file.file.close()
    res = transform_predict_csv(df)
    return res
