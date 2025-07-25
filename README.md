## Income Classification with XGBoost and PCA

This project applies machine learning techniques to predict whether an individual's income exceeds \$50K/year based on census data. It uses the **XGBoost** classifier and includes **dimensionality reduction with PCA** for visualization.

---

### Dataset

* **Name:** Adult Income Dataset
* **File Used:** `adult_moins.csv`
* **Target Variable:** `income` (binary: `>50K` or `<=50K`)

---

### Objectives

1. Clean and preprocess census data.
2. Train an XGBoost classifier on the full feature set.
3. Evaluate the model performance.
4. Apply PCA for dimensionality reduction to 2D.
5. Visualize decision boundaries of XGBoost on reduced data.

---

### Requirements

The notebook uses the following Python libraries:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

You can install them using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

### Steps Performed

#### 1. **Importing Libraries**

All required libraries for data manipulation, visualization, modeling, and evaluation are imported.

#### 2. **Data Loading and Cleaning**

* Columns are assigned manually.
* Missing values represented as `" ?"` are removed.
* Categorical features are encoded using one-hot encoding.
* The target variable is encoded to binary (1 for `>50K`, 0 for `<=50K`).

#### 3. **Feature Scaling and Splitting**

* Features are standardized using `StandardScaler`.
* The dataset is split into training and testing sets (80/20 split).

#### 4. **Training XGBoost on Full Data**

* An XGBoost classifier is trained using all the original dimensions.
* Model performance is evaluated using accuracy, classification report, and confusion matrix.

#### 5. **Dimensionality Reduction with PCA**

* PCA reduces the high-dimensional feature space to 2 components.
* The reduced data is split into training and testing sets for visualization.

#### 6. **Training XGBoost on PCA Data**

* A second XGBoost model is trained using only the 2D PCA-transformed data.

#### 7. **Visualization**

* A scatter plot is used to visualize the test data in 2D PCA space.
* A decision boundary is overlaid to show how the XGBoost classifier separates classes.

---

### Outputs

* Classification metrics (precision, recall, f1-score)
* Accuracy score
* Confusion matrix heatmap
* 2D decision boundary plot (XGBoost + PCA)

---

### Notes

* PCA is only used for **visualization**, not for improving model accuracy.
* XGBoost performs better with well-preprocessed data — this example showcases both power and visualization.
