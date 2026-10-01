# 📧 Email / SMS Spam Detection with Machine Learning

A Natural Language Processing (NLP) project that classifies messages as **spam** or **ham** (legitimate) using TF-IDF features and classic machine learning models.

Built as **Task 4** of the Oasis Infobyte (OIBSIP) Data Science internship.

---

## 📌 Objective

Build a binary text classifier that separates spam from legitimate messages, and evaluate it with metrics that suit an imbalanced dataset (accuracy, precision, recall, F1-score, confusion matrix).

## 📂 Dataset

**SMS Spam Collection** (UCI Machine Learning Repository / Kaggle)

- 5,572 English SMS messages labeled `ham` or `spam`
- 403 duplicate rows removed, leaving **5,169** unique messages
- Class distribution after cleaning: **87.4% ham (4,516)** and **12.6% spam (653)**, so the data is imbalanced and accuracy alone can mislead

Download `spam.csv` from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset) and place it in the project folder.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · NLTK · matplotlib · seaborn · WordCloud · Jupyter Notebook

## 🔄 Workflow

1. **Data loading and class distribution check** (counts, percentages, charts)
2. **Text preprocessing:** lowercase → remove URLs, punctuation, digits and non-letter characters → stopword removal → Porter stemming
3. **Feature extraction:** TF-IDF Vectorizer (5,000 features, unigrams + bigrams), fitted on training data only to avoid leakage
4. **Train/test split:** 80/20, stratified to keep the spam ratio equal in both sets
5. **Model training:** Multinomial Naive Bayes, Logistic Regression, Linear SVM
6. **Evaluation:** accuracy, precision, recall, F1-score, classification reports, confusion matrices
7. **Bonus:** WordCloud visualisations for spam and ham, top spam-indicating words, live prediction on new messages

## 📊 Results (test set: 1,034 messages)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Multinomial Naive Bayes | 0.9816 | 0.9746 | 0.8779 | 0.9237 |
| Logistic Regression | 0.9729 | 0.8815 | 0.9084 | 0.8947 |
| Linear SVM | 0.9807 | 0.9370 | 0.9084 | 0.9225 |

**Key takeaways**

- Naive Bayes has the highest precision (few genuine messages wrongly flagged) but the lowest recall (more spam slips through).
- Logistic Regression and Linear SVM catch more spam (recall ≈ 0.91). Linear SVM gives the best balance, with an F1 very close to Naive Bayes.
- **Why recall matters:** a missed spam message (false negative) can be a phishing link or scam that reaches the user. But recall alone isn't enough, since flagging everything as spam gives perfect recall and a useless filter. That is why precision and F1 are reported too.

Top spam-indicating words learned by the model include: `txt`, `call`, `free`, `text`, `mobil`, `repli`, `claim`, `prize`, `stop`.

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter

# 3. Add spam.csv to the project folder, then launch the notebook
jupyter notebook Email_Spam_Detection.ipynb
```

The notebook downloads the NLTK stopwords list automatically on first run.

## 📁 Project Structure

```
├── Email_Spam_Detection.ipynb   # Full commented notebook
├── spam.csv                     # Dataset (download from Kaggle)
└── README.md
```

## 🔮 Future Improvements

- Cross-validation and hyperparameter tuning with `GridSearchCV`
- Lemmatization instead of stemming
- Handling class imbalance (SMOTE or class weights across all models)
- Testing on a real email dataset
- Deploying as a simple web app (Streamlit or Flask)

## 👤 Author

**Karan Tormal**
[LinkedIn](https://www.linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

## 🙏 Acknowledgements

- Dataset: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* ACM Symposium on Document Engineering, 2011.
- Internship: [Oasis Infobyte](https://oasisinfobyte.com)
