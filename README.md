# Data-Mining-Exercises
This repository contains a course project focused on key tasks in data mining: classification, clustering, and frequent pattern mining.

## Algorithms Implemented
Classification:
- Decision Trees (Gain Ratio & Gini Index)
- Naïve Bayes
- Neural Networks (1 and 2 hidden layers)
- Support Vector Machines (SVM)

Clustering:
- AGNES (Agglomerative Hierarchical Clustering)
- DBSCAN
- K-Means

Frequent Pattern Mining:
- Apriori Algorithm
- FP-Growth Algorithm

## Datasets Used
- Bank Customer Dataset — used for clustering tasks
- US Weather Events Dataset — used for classification and pattern mining

Each task is implemented from scratch or using standard libraries to provide hands-on understanding of the underlying techniques. Visualizations, confusion matrices, and evaluation metrics are included where applicable. 
Both datasets required significant preprocessing to ensure meaningful analysis:

US Weather Events Dataset:
- The raw dataset contained a large proportion of irrelevant or redundant fields.
- Many features were noisy or sparse, necessitating removal or transformation.
- New, meaningful features were engineered to support effective classification and pattern mining—these included aggregated event categories, temporal features, and region-based groupings.

Bank Customer Dataset:
- Preprocessing involved handling missing values, normalizing numerical features, and encoding categorical variables for clustering tasks.
- These preprocessing steps were essential to improve model performance and ensure the relevance of extracted patterns.

## Example Results:
Clustering:

![image](https://github.com/user-attachments/assets/a75e1f74-d57b-4c44-afc6-7837da5c47c1)

Frequent Pattern Mining:

![image](https://github.com/user-attachments/assets/e111b7c9-718c-4870-b1b6-771c26c95974)

Classification:

| Model                             | TN   | FP   | FN   | TP    | Accuracy | Precision | Recall | F1 Score |
|----------------------------------|------|------|------|-------|----------|-----------|--------|----------|
| Decision Tree Gain Ratio         | 8920 | 2045 | 1146 | 9818  | 0.8545   | 0.8276    | 0.8955 | 0.8602   |
| Gain Ratio + Bagging             | 9093 | 1872 | 933  | 10031 | 0.8721   | 0.8427    | 0.9149 | 0.8773   |
| Gain Ratio + Boosting            | 9316 | 1649 | 1091 | 9873  | 0.8751   | 0.8569    | 0.9005 | 0.8781   |
| Decision Tree Gini Index         | 8926 | 2039 | 1151 | 9813  | 0.8545   | 0.8280    | 0.8950 | 0.8602   |
| Gini Index + Bagging             | 9077 | 1888 | 913  | 10051 | 0.8723   | 0.8419    | 0.9167 | 0.8777   |
| Gini Index + Boosting            | 9299 | 1666 | 1102 | 9862  | 0.8738   | 0.8555    | 0.8995 | 0.8769   |
| Gini Index CV Avg                | –    | –    | –    | –     | 0.8583   | 0.8308    | 0.9001 | 0.8639   |
| Gini + Bagging CV Avg            | –    | –    | –    | –     | 0.8755   | 0.8528    | 0.9076 | 0.8794   |
| Gini + Boosting CV Avg           | –    | –    | –    | –     | 0.8777   | 0.8586    | 0.9044 | 0.8809   |
| Naive Bayes                      | 6698 | 4267 | 2050 | 8914  | 0.7119   | 0.6763    | 0.8130 | 0.7384   |
| Naive Bayes + Bagging            | 6820 | 4145 | 2134 | 8830  | 0.7137   | 0.6805    | 0.8054 | 0.7377   |
| Naive Bayes + Boosting           | 6698 | 4267 | 2050 | 8914  | 0.7119   | 0.6763    | 0.8130 | 0.7384   |
| Neural Net (1 HL)                | 9113 | 1851 | 2083 | 8882  | 0.8206   | 0.8275    | 0.8100 | 0.8187   |
| (1 HL) + Bagging                 | 9210 | 1754 | 2024 | 8941  | 0.8277   | 0.8360    | 0.8154 | 0.8256   |
| (1 HL) + Boosting                | –    | –    | –    | –     | –        | –         | –      | –        |
| Neural Net (2 HL)                | 9162 | 1802 | 2072 | 8893  | 0.8233   | 0.8315    | 0.8110 | 0.8211   |
| (2 HL) + Bagging                 | 9212 | 1752 | 2045 | 8920  | 0.8269   | 0.8358    | 0.8135 | 0.8245   |
| (2 HL) + Boosting                | –    | –    | –    | –     | –        | –         | –      | –        |
| Support Vector Machine           | 9342 | 1623 | 2215 | 8749  | 0.8250   | 0.8435    | 0.7980 | 0.8201   |
| SVM + Bagging                    | 9259 | 1706 | 2134 | 8830  | 0.8249   | 0.8381    | 0.8054 | 0.8214   |
| SVM + Boosting                   | –    | –    | –    | –     | –        | –         | –      | –        |
