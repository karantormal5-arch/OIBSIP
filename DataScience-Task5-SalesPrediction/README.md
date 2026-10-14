# Sales Prediction Using Python

A regression project that predicts product sales from advertising spend on three media channels: **TV, Radio and Newspaper**. A Linear Regression model is trained as the baseline.

## Dataset

The **Advertising** dataset (from *An Introduction to Statistical Learning*) covers 200 markets.

| Column | Description |
|---|---|
| `TV` | Advertising spend on TV ($ thousands) |
| `Radio` | Advertising spend on Radio ($ thousands) |
| `Newspaper` | Advertising spend on Newspaper ($ thousands) |
| `Sales` | Product sales, the target (thousand units) |

The raw CSV also has an unnamed first column that only stores the row index. The notebook drops it. The data has no missing values and no duplicate rows.

## Project Structure

```
.
├── Sales_Prediction.ipynb   # full analysis with outputs
├── Advertising.csv          # dataset
└── README.md
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## What the Notebook Does

1. Loads the data and runs EDA: null check, descriptive statistics and a pairplot of all features
2. Plots Sales vs TV, Sales vs Radio and Sales vs Newspaper as individual scatter plots
3. Draws a correlation matrix heatmap
4. Splits the data into 80% train and 20% test (`random_state=42`)
5. Trains a Linear Regression baseline and evaluates it with R², MAE, MSE and RMSE
6. Plots actual vs predicted sales on the test set

## Results

**Model coefficients** (intercept = 2.9791)

| Feature | Coefficient |
|---|---|
| TV | 0.0447 |
| Radio | 0.1892 |
| Newspaper | 0.0028 |

**Performance**

| Metric | Train | Test |
|---|---|---|
| R² | 0.8957 | 0.8994 |
| MAE | 1.1985 | 1.4608 |
| MSE | 2.7051 | 3.1741 |
| RMSE | 1.6447 | 1.7816 |

## Key Findings

- Sales is most correlated with **TV (r = 0.78)**, then **Radio (r = 0.58)**, then **Newspaper (r = 0.23)**.
- The three spend columns are only weakly correlated with each other, so multicollinearity is not a serious concern.
- The baseline model explains about **90% of the variance in sales** on unseen data (test R² ≈ 0.90, RMSE ≈ 1.78 thousand units, against average sales of about 14.0).
- Train and test scores are close, so the model does not appear to overfit.
- Per dollar spent, **Radio** adds the most sales, **TV** is the strongest driver overall, and the **Newspaper** coefficient is close to zero.
- The test set has only 40 rows, so exact numbers can shift a little with a different random split.

## How to Run

1. Install the requirements:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

2. Keep `Advertising.csv` in the same folder as the notebook.
3. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook Sales_Prediction.ipynb
   ```

4. Choose **Kernel > Restart & Run All**.

## Possible Improvements

- Use cross-validation for a more stable performance estimate
- Check whether dropping Newspaper changes the results
- Try interaction or polynomial features (for example TV × Radio)
- Compare against tree-based models such as Random Forest
