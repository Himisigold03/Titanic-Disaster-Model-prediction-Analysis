# Titanic-Disaster-Model-prediction-Analysis
This is a model built to predict the survival of an individual in a titanic diasater

---

````markdown
# Titanic Survival Prediction  
A complete machine learning pipeline that predicts passenger survival on the Titanic using multiple classification algorithms.  
This project covers **data loading, cleaning, visualization, preprocessing, feature engineering, model training, evaluation, and model saving**.

---

## Project Overview  
The goal of this project is to build predictive models to determine whether a passenger survived the Titanic disaster based on features such as age, sex, ticket class, fare, and more.

The workflow includes:
- Importing libraries  
- Loading and exploring the dataset  
- Handling missing values  
- Encoding categorical variables  
- Building multiple ML models  
- Comparing model performance  
- Saving the best model  

---

## Dataset  
The dataset used is the **Kaggle Titanic "train.csv"** file.  
It contains 891 rows and 12 columns.

### Sample Data
| PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
|-------------|----------|--------|------|-----|-----|--------|--------|--------|-------|--------|----------|
| 1 | 0 | 3 | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.25 | NaN | S |

---

## Data Preprocessing  
Key preprocessing steps include:

- Dropped unnecessary columns: **Name, Cabin, Ticket, PassengerId**
- Filled missing:
  - `Embarked` → mode  
  - `Age` → mean  
- Encoded `Sex` and `Embarked` using **LabelEncoder**
- Removed duplicate names

---

## Exploratory Data Analysis (EDA)

### Visualizations include:
- Survival counts  
- Gender distribution  
- Embarkation location distribution  

Example code:

```python
data_df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.show()


## Machine Learning Models

The following models were trained and evaluated:

| Model                     | Accuracy   |
| ------------------------- | ---------- |
| **Logistic Regression**   | **0.8134** |
| Decision Tree             | 0.7388     |
| Random Forest             | 0.7910     |
| K-Nearest Neighbors       | 0.7052     |
| Support Vector Classifier | 0.6567     |
| Gradient Boosting         | 0.7985     |
| XGBoost                   | 0.7687     |

### Best Model: **Logistic Regression (0.8134 accuracy)**


## Model Evaluation Function

```python
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if set(y) == {0, 1}:
        acc = metrics.accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {acc:.4f}")
        print(metrics.classification_report(y_test, y_pred))




## Saving the Model

```python
import joblib

joblib.dump(LogisticRegression, "best_model.pkl")
joblib.dump(LEC_Sex, "LEC_Sex.pkl")
joblib.dump(LEC_Embarked, "LEC_Embarked.pkl")


Model and encoders are saved for reuse in deployment or integration.


## 🛠 Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-Learn
* XGBoost
* Joblib

## How to Run the Project

```bash
git clone <your-repo-url>
cd your-repo-folder

# Install dependencies
pip install -r requirements.txt

# Run the script
python titanic_prediction.py


## Conclusion

This project demonstrates the full end-to-end process of:

* Cleaning real-world data
* Visualizing relationships
* Engineering features
* Training multiple machine learning models
* Evaluating performance
* Saving the best model for deployment

It provides a solid foundation for further improvements such as hyperparameter tuning, cross-validation, model deployment (Flask/FastAPI), or a web UI.


## If you found this project helpful

Consider giving it a **star** 🌟 on GitHub!
