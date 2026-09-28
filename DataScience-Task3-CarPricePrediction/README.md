# 🚗 Car Price Prediction with Machine Learning

**OIBSIP – Data Science Internship – Task 3**

## 📌 Objective
Build a regression model that predicts the selling price of a used car from features such as brand, age, mileage, fuel type and transmission.

## 📂 Dataset
[Vehicle dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (Kaggle), file `Car_details_v3.csv`: 8,128 listings with name, year, selling price, km driven, fuel, seller type, transmission, owner, mileage, engine, max power, torque and seats.

## 🛠️ Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## 🔄 Workflow
1. **Data cleaning**: removed 1,202 duplicate rows and 209 rows with nulls; stripped units from `mileage`, `engine` and `max_power`; standardised categorical text (e.g. "petrol" vs "Petrol").
2. **Feature engineering**: `car_age` from the year column; `brand` extracted from the car name.
3. **EDA**: selling price distribution, price vs fuel type (box plot), price vs car age (scatter plot).
4. **Encoding**: one-hot encoding of fuel, seller type, transmission, owner and brand.
5. **Correlation heatmap** of the numeric features.
6. **Train/test split**: 80/20 (5,373 train rows, 1,344 test rows).
7. **Models**: Linear Regression (baseline) and Random Forest Regressor.
8. **Evaluation**: MAE, RMSE and R².
9. **Feature importance** from the Random Forest.

## 📊 Results

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| Linear Regression | 131,566 | 238,060 | 0.742 |
| **Random Forest Regressor** | **73,855** | **124,848** | **0.929** |

The Random Forest is the best model, since it captures non-linear interactions between features (for example brand, age and engine power together).

## 🔍 Key Insights
- Selling price is heavily right-skewed: most cars are cheap, with a long tail of luxury cars.
- Diesel cars have the highest median price (about ₹525,000, versus about ₹320,000 for petrol).
- `max_power` is the strongest predictor of price (correlation 0.69), followed by `engine` (0.44).
- `car_age` is negatively correlated with price (−0.43).
- In the Random Forest, `max_power` (about 0.58) and `car_age` (about 0.25) dominate the feature importances.

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Car_Price_Prediction.ipynb
```
Keep `Car_details_v3.csv` in the same folder as the notebook.

## 🚀 Possible Improvements
- Gradient Boosting / XGBoost
- Hyperparameter tuning with GridSearchCV
- Log-transforming the skewed target variable
