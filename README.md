# Curso CODER - DataScience-II

Predicción de Precios de Propiedades en California
Este proyecto de Data Science tiene como objetivo construir un modelo de Machine Learning para predecir el precio de propiedades en California. A partir de un análisis exhaustivo de datos y el uso de múltiples algoritmos de regresión, se logró identificar los factores clave que determinan el precio de las viviendas y seleccionar el modelo más eficiente para esta tarea.

## 📊 Contenido del Proyecto

EDA (Exploratory Data Analysis): Visualización y análisis de las principales características del dataset.
Limpieza de Datos: Tratamiento de valores atípicos mediante IsolationForest.
Transformaciones: Encoding de variables categóricas, escalamiento de datos y PCA.
Modelado: Evaluación de modelos como KNN, Linear Regression y XGBoost.
Optimización: Ajuste de hiperparámetros usando Halving Grid Search y validación cruzada.
Resultados: Evaluación mediante métricas como R² y MSE.

## ⚙️ Requisitos de Instalación

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes librerías:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lazypredict

## 📚 Librerías Utilizadas

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler  
from sklearn.model_selection import train_test_split, cross_val_score, HalvingGridSearchCV  
from sklearn.decomposition import PCA  
from sklearn.linear_model import LinearRegression  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import mean_squared_error, r2_score  
from xgboost import XGBRegressor  
from lazypredict.Supervised import LazyRegressor

## 🚀 Ejecución
### Clona este repositorio:

git clone <https://github.com/djender985/>
cd <CursoDataScience-II>

### Instala las dependencias necesarias:

pip install -r requirements.txt

Abre y ejecuta el Jupyter Notebook para reproducir el análisis y entrenamiento de modelos.


## 📈 Resultados
El modelo más eficiente fue XGBoost, alcanzando un R² de 0.84 después de la optimización de hiperparámetros.
