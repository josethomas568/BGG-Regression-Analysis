# Board Game Rating Prediction Engine

A robust machine learning pipeline designed to predict the average community rating of tabletop games using the BoardGameGeek dataset. 

By implementing ensemble methods using gradient-boosted trees, this engine effectively captures complex non-linear relationships and feature interactions to drive a massive **93.6% improvement in predictive accuracy ($R^2$)** over baseline regression models (up from an initial 30.6% improvement with regularized linear models).

## The Business Problem
When designing and publishing new tabletop games, creators need to understand exactly what attributes—mechanics, complexity, playtime, and themes—resonate most with the gaming community. 

Traditional linear models fail to capture the complex, compounding interactions between various game mechanics. They struggle to accurately map how nuanced characteristics (e.g., how "Area Majority" combined with high "Complexity") influence the final community rating.

## Feature Engineering Strategy
To force the model to learn the true value of specific game traits, the architecture was designed to process a rich set of 39 engineered features. 

By scaling numerical distributions and one-hot encoding categorical variables, the gradient-boosted trees dedicated 100% of their computational power to evaluating how specific mechanics, domains (e.g., Strategy Games, Party Games), complexity weights, and user engagement metrics drive the final community rating. 

## Model Performance
Trained on a 20,322-record dataset from BoardGameGeek, the models were evaluated using 5-fold cross-validation to guarantee the reliability of the average rating predictions.

| Model | Strategy | $R^2$ Score | Improvement |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Baseline | 0.4763 | Baseline |
| **Ridge Regression** | L2 Regularization | 0.6222 | +30.63% |
| **Random Forest** | Bagging Ensemble | 0.8079 | +69.62% |
| **Gradient Boosting** | Boosting Ensemble | **0.9221** | **+93.59%** |

---

## Tech Stack & Architecture
* Data Engineering & EDA: `pandas`, `numpy`, `seaborn`, `matplotlib`
* Machine Learning: `scikit-learn` (HistGradientBoosting, RandomForest, Ridge, Lasso)
* Data Processing: `StandardScaler`, `SimpleImputer`

---

Developed by Jose Thomas as a demonstration of end-to-end data science, exploratory data analysis, and advanced predictive modeling.
