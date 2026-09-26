Car Price Prediction with Machine Learning

OIBSIP - Data Science Internship - Task 3

📌 Objective

Build a regression model that predicts the selling price of a used car based on features such as brand, age, mileage, fuel type, and transmission.

🛠️ Tech Stack
Python
pandas, numpy
scikit-learn
matplotlib, seaborn
Jupyter Notebook
📂 Dataset

Vehicle Dataset from CarDekho (Kaggle) — Car_details_v3.csv

🔍 Approach
Loaded and cleaned the dataset (removed duplicates, handled nulls, stripped units from numeric columns)
Feature engineering — created car_age from year, extracted brand from car name
Performed Exploratory Data Analysis (price distribution, price vs fuel type, price vs car age)
Encoded categorical variables using One-Hot Encoding
Built a correlation heatmap to understand feature relationships
Trained and compared two regression models: Linear Regression and Random Forest Regressor
Evaluated both models using MAE, RMSE, and R² score
Visualized feature importance for the best-performing model
📊 Results
Model	MAE	RMSE	R² Score
Linear Regression	139,224.46	224,710.83	0.7699
Random Forest Regressor	73,854.75	124,847.54	0.9290

Best Model: Random Forest Regressor performed better, as it captures non-linear relationships between features (like engine power, age, and brand) better than a simple linear model.

Top Predictive Features: Max Power, Car Age, Engine Size

📁 Files in this folder
Car_Price_Prediction.ipynb — Full notebook with code, EDA, model training and evaluation
Car_details_v3.csv — Dataset used for training
🎥 Demo Video

[Link to LinkedIn demo video]

Part of the Oasis Infobyte Summer Internship Program (OIBSIP)

Content

PDF

car data.csv

CSV

CAR DETAILS FROM CAR DEKHO.csv

CSV

Car details v3.csv

CSV

car details v4.csv

CSV

Car_Price_Prediction.ipynb

IPYNB
