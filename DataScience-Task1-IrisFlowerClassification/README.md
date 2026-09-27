# Iris Flower Classification

**Author:** Karan Dattatray Tormal
**Task:** OIBSIP Data Science – Task 1

## Objective

Train a machine learning classification model to identify the species of an iris flower
(*Setosa*, *Versicolor*, or *Virginica*) from four physical measurements: sepal length,
sepal width, petal length, and petal width.

## Dataset

The Iris dataset is loaded directly from `sklearn.datasets.load_iris()` — no external
download required. It contains 150 samples, 50 for each of the three species, with no
missing values.

| Feature | Description |
|---|---|
| sepal length (cm) | Length of the sepal |
| sepal width (cm) | Width of the sepal |
| petal length (cm) | Length of the petal |
| petal width (cm) | Width of the petal |
| species | Target class: Setosa / Versicolor / Virginica |

## Tech Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

## Project Structure

```
├── Iris_Flower_Classification_Final.ipynb   # Main notebook
└── README.md                                # This file
```

## Workflow

1. **Load Data** – Load the Iris dataset into a pandas DataFrame.
2. **EDA** – Check shape, data types, missing values, and descriptive statistics.
3. **Visualization** – Pairplot, per-feature boxplots by species, and a petal
   length vs. petal width scatterplot.
4. **Feature Selection** – ANOVA F-test to identify the most discriminative
   features (petal length and petal width came out on top).
5. **Train/Test Split** – 80/20 stratified split.
6. **Feature Scaling** – `StandardScaler`, fit on the training set only.
7. **Model Training** – Four classifiers trained and compared:
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - Random Forest
8. **Evaluation** – Accuracy, confusion matrix, and full classification report
   (precision, recall, F1-score) for every model.
9. **Best Model Selection** – Models ranked by test accuracy.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.9333 |
| K-Nearest Neighbors | 0.9333 |
| Decision Tree | 0.9333 |
| Random Forest | 0.9000 |

**Best model:** Logistic Regression (0.9333 accuracy on the held-out test set).

Petal length and petal width were confirmed — both visually (boxplots, scatterplot)
and statistically (ANOVA F-test) — to be the most discriminative features for
separating the three species.

## How to Run

1. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Iris_Flower_Classification_Final.ipynb
   ```
3. Run all cells top to bottom.

## Conclusion

All four models perform well on the Iris dataset because the classes are largely
well-separated, particularly on petal measurements. Logistic Regression was selected
as the best-performing model on this train/test split, tying with KNN and Decision
Tree at 93.3% accuracy while remaining the simplest and most interpretable of the three.
