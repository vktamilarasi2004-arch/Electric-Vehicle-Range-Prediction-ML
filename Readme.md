import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
## 🖥️ Output Page

The output page provides an interactive interface where users can enter electric vehicle details and obtain the predicted driving range.

### 🔹 Output Interface Description
- The user selects or enters EV specifications through input fields.
- Inputs include battery capacity, vehicle efficiency, model year, and other parameters.
- After clicking the **Predict Range** button, the trained Machine Learning model processes the data.
- The system displays the **predicted EV driving range in kilometers (km)** instantly.

### 🔹 Sample Input
- Battery Capacity: 40 kWh  
- Vehicle Efficiency: 0.15 kWh/km  
- Model Year: 2022  

### 🔹 Sample Output
Predicted EV Range: 260 km
### 🔹 Output Features
- Simple and user-friendly interface
- Fast and accurate prediction
- Clear display of predicted range
- Helps reduce range anxiety for EV users

### 🔹 Result
The system successfully predicts the driving range of an electric vehicle using Machine Learning techniques, demonstrating effective application of data-driven models in the EV domain.
