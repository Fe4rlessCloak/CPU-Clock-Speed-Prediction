from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
import numpy as np
from typing import Literal
import pandas as pd
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from pydantic.json_schema import SkipJsonSchema

class BaseCPU(BaseModel):
    Lithography:int = Field(gt=0)
    Cores:int = Field(gt=0)
    Power:float = Field(gt=0)
    ReleaseDate: int = Field(gt=2000)
    L2_per_Core_KB: int = Field(gt=0)
    Is_Tiled: bool 
    Is_Mesh: bool
    Node_Density: float = Field(gt=0)
    Node_Maturity_Years: float = Field(gt=-1)
    P_Cores: int = 0
    Threads: int = 0
    Family: Literal["Celeron", "Core", "Core Ultra", "Intel", "Pentium"]
    Tier: Literal["Embedded", "Extreme_Low_Power", "High_Perf", "High_Perf_Mobile", "Mobile_Legacy", "No_Graphics", "Power_Optimized", "Standard", "Standard_Graphics", "Ultra_Low_Power"]

    Power_Starvation_Index: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    Cores_x_Is_Mesh: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    TDP_x_Is_Tiled: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    P_Core_Ratio: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    TDP_per_PCore: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    Interaction_Power_PCore: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    Legacy_Node_Penalty: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    TDP_per_Core: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    Threads_per_Core: SkipJsonSchema[float] = Field(default=0.0, exclude=True)
    Log_Node_Density: SkipJsonSchema[float] = Field(default=0.0, exclude=True)

    fam_Celeron: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    fam_Core: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    fam_Core_Ultra: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    fam_Intel: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    fam_Pentium: SkipJsonSchema[bool] = Field(default=False, exclude=True)

    tier_Embedded: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_Extreme_Low_Power: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_High_Perf: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_High_Perf_Mobile: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_Mobile_Legacy: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_No_Graphics: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_Power_Optimized: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_Standard: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_Standard_Graphics: SkipJsonSchema[bool] = Field(default=False, exclude=True)
    tier_Ultra_Low_Power: SkipJsonSchema[bool] = Field(default=False, exclude=True)

    def calculate_engineered_features(self):
        self.Cores_x_Is_Mesh = float(self.Cores if self.Is_Mesh else 0.0)
        self.TDP_x_Is_Tiled = float(self.Power if self.Is_Tiled else 0.0)
        self.TDP_per_Core = self.Power / self.Cores
        self.Threads_per_Core = self.Threads / self.Cores
        self.Power_Starvation_Index = (self.Cores ** 2) / self.Power

        self.P_Core_Ratio = self.P_Cores / self.Cores
        self.TDP_per_PCore = self.Power / self.P_Cores if self.P_Cores > 0 else 0.0
        self.Interaction_Power_PCore = self.TDP_per_PCore * self.P_Core_Ratio
        self.Legacy_Node_Penalty = self.Lithography * self.Node_Maturity_Years
        self.Log_Node_Density = np.log1p(self.Node_Density)
  
        match self.Family:
            case "Celeron":
                self.fam_Celeron = True
            case "Core":
                self.fam_Core = True
            case "Core Ultra":
                self.fam_Core_Ultra = True
            case "Intel":
                self.fam_Intel = True
            case "Pentium":
                self.fam_Pentium = True

        match self.Tier:
            case "Embedded":
                self.tier_Embedded = True
            case "Extreme_Low_Power":
                self.tier_Extreme_Low_Power = True
            case "High_Perf":
                self.tier_High_Perf = True
            case "High_Perf_Mobile":
                self.tier_High_Perf_Mobile = True
            case "Mobile_Legacy":
                self.tier_Mobile_Legacy = True
            case "No_Graphics":
                self.tier_No_Graphics = True
            case "Power_Optimized":
                self.tier_Power_Optimized = True
            case "Standard":
                self.tier_Standard = True
            case "Standard_Graphics":
                self.tier_Standard_Graphics = True
            case "Ultra_Low_Power":
                self.tier_Ultra_Low_Power = True
    def getDictionary(self):
        data = self.__dict__
        name_translator = {
            'Lithography': 'Lithography(nm)',
            'Power': 'TDP(W)',
            'ReleaseDate': 'Release Date',
            'tier_Extreme_Low_Power': 'tier_Extreme Low Power',
            'tier_High_Perf': 'tier_High Perf',
            'tier_High_Perf_Mobile': 'tier_High Perf Mobile',
            'tier_Mobile_Legacy': 'tier_Mobile (Legacy)',
            'tier_No_Graphics': 'tier_No Graphics',
            'tier_Power_Optimized': 'tier_Power Optimized',
            'tier_Standard_Graphics': 'tier_Standard / Graphics',
            'tier_Ultra_Low_Power': 'tier_Ultra-Low Power'
        } 
        translated_dict = {}
        for key, value in data.items():
            notebook_key = name_translator.get(key, key)
            translated_dict[notebook_key] = value
        return translated_dict
    def getDataFrame(self, translated_dict):
        df_inference = pd.DataFrame([translated_dict])
        df_inference = df_inference[features]
        return df_inference
    
scaler: StandardScaler = None
poly_transformer: PolynomialFeatures = None
model: Ridge = None

def loadWeights():
    weightsDirectory = os.path.join(os.path.dirname(__file__), "weights")
    global scaler, poly_transformer, model
    try:
        scaler = joblib.load(os.path.join(weightsDirectory,"cpu_scaler.joblib"))
        poly_transformer = joblib.load(os.path.join(weightsDirectory, "cpu_poly_transformer.joblib"))
        model = joblib.load(os.path.join(weightsDirectory, "cpu_ridge_model.joblib"))
    except Exception as e:
        print(f"Failed to load weights: {e}")
loadWeights() 


app = FastAPI(
    title="CPU Clock Speed Predictor API",
    description="Backend API serving a scikit-learn model to predict CPU frequencies",
    version="1.0.0"
)


features = ['Lithography(nm)', 'Cores', 'TDP(W)', 'Release Date', 'L2_per_Core_KB', 'Is_Tiled', 'Is_Mesh', 'Log_Node_Density', 'Cores_x_Is_Mesh', 'TDP_x_Is_Tiled', 'Node_Maturity_Years', 'TDP_per_Core', 'Threads_per_Core', 'Power_Starvation_Index', 'P_Core_Ratio', 'TDP_per_PCore', 'Interaction_Power_PCore', 'Legacy_Node_Penalty', 'fam_Celeron', 'fam_Core', 'fam_Core_Ultra', 'fam_Intel', 'fam_Pentium', 'tier_Embedded', 'tier_Extreme Low Power', 'tier_High Perf', 'tier_High Perf Mobile', 'tier_Mobile (Legacy)', 'tier_No Graphics', 'tier_Power Optimized', 'tier_Standard', 'tier_Standard / Graphics', 'tier_Ultra-Low Power']




@app.get("/")
def greet():
    return "Good Day Sire"

@app.post("/predict")
def getSpecifications(CPU:BaseCPU):
    CPU.calculate_engineered_features()
    cpu_dict = CPU.getDictionary()
    df_inference = CPU.getDataFrame(cpu_dict)

    poly_data = poly_transformer.transform(df_inference)
    scaled_data = scaler.transform(poly_data)
    
  

    predicted_array = model.predict(scaled_data)

 
    final_clock_speed = float(predicted_array[0])

    return {
    "status": "success",
    "predicted_turbo_frequency_ghz": round(final_clock_speed, 3),
    "predicted_turbo_frequency_mhz": int(final_clock_speed * 1000)
    }