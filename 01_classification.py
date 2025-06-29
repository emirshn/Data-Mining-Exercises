import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# --- Preprocessing ---
def preprocess_data():
    df = pd.read_csv("bank-additional-full.csv", sep=';')
    # Drop duration column because it effects output highly and knowing duration before a call is impossible.
    # For realistic prediction this should be discarded. (From dataset readme file)
    # Others are dropped because they are not useful for prediction.
    drop_cols = [
        "duration",              
        "emp.var.rate",
        "cons.price.idx",       
        "cons.conf.idx",         
        "euribor3m",             
        "nr.employed"            
    ]
    df.drop(columns=drop_cols, inplace=True)

    # Convert target 'y' from 'yes'/'no' to 1/0
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # Handle missing data as class label (From dataset readme file)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if "y" in categorical_cols:
        categorical_cols.remove("y")
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # Bin age and pdays
    df['age_group'] = pd.cut(
          df['age'],
          bins=[0, 30, 60, 100],
          labels=['young', 'middle-aged', 'senior']
      )
    df.drop(columns=['age'], inplace=True)
    categorical_cols.append('age_group')
    # 999 represents not contacted instead of  999 day
    df['pdays_status'] = df['pdays'].apply(
      lambda x: 'not_contacted' if x == 999 else (
        'recent' if x <= 10 else 'older'
      )
    )
    categorical_cols.append('pdays_status')
    df.drop(columns=['pdays'], inplace=True)

    # One-hot encode categorical attributes
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop("y", axis=1)
    y = df_encoded["y"]

    # SMOTE for oversampling
    X, y = SMOTE(random_state=50).fit_resample(X, y)

    return X, y

X, y = preprocess_data()

# --- Evaluation ---
def evaluate_model(model, X_train, X_test, y_train, y_test, label="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{label} Evaluation:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall:    {report['1']['recall']:.4f}")
    print(f"F1 Score:  {report['1']['f1-score']:.4f}")

def evaluate_model_cv(model, X, y, label="Model", cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        accuracies.append(report["accuracy"])
        precisions.append(report['1']['precision'])
        recalls.append(report['1']['recall'])
        f1s.append(report['1']['f1-score'])

    print(f"\n{label} Cross-Validation ({cv} folds) Average:")
    print(f"Accuracy:  {sum(accuracies)/cv:.4f}")
    print(f"Precision: {sum(precisions)/cv:.4f}")
    print(f"Recall:    {sum(recalls)/cv:.4f}")
    print(f"F1 Score:  {sum(f1s)/cv:.4f}")

# --- Classifiers ---
def run_decision_tree(criterion):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    base = DecisionTreeClassifier(criterion=criterion, class_weight='balanced', random_state=42)
    bag = BaggingClassifier(estimator=base, n_estimators=50, random_state=42, n_jobs=-1) # n_jobs run it parallel instead of using one cpu
    boost = AdaBoostClassifier(estimator=base, n_estimators=50, random_state=42)

    name = "Gain Ratio" if criterion == 'entropy' else "Gini Index"
    evaluate_model(base, X_train, X_test, y_train, y_test, f"Decision Tree {name}")
    evaluate_model(bag, X_train, X_test, y_train, y_test, f"Decision Tree {name} Bagging")
    evaluate_model(boost, X_train, X_test, y_train, y_test, f"Decision Tree {name} Boosting")

def run_decision_tree_cv(criterion, cv=5):
    base = DecisionTreeClassifier(criterion=criterion, class_weight='balanced', random_state=42)
    bag = BaggingClassifier(estimator=base, n_estimators=50, random_state=42, n_jobs=-1)
    boost = AdaBoostClassifier(estimator=base, n_estimators=50, random_state=42)

    name = "Gain Ratio" if criterion == 'entropy' else "Gini Index"
    evaluate_model_cv(base, X, y, f"Decision Tree {name}", cv)
    evaluate_model_cv(bag, X, y, f"Decision Tree {name} Bagging", cv)
    evaluate_model_cv(boost, X, y, f"Decision Tree {name} Boosting", cv)

def run_naive_bayes():
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=43)
    base = GaussianNB()
    bag = BaggingClassifier(estimator=base, n_estimators=50, random_state=43, n_jobs=-1)
    boost = AdaBoostClassifier(estimator=base, n_estimators=50, random_state=43)

    evaluate_model(base, X_train, X_test, y_train, y_test, "Naive Bayes")
    evaluate_model(bag, X_train, X_test, y_train, y_test, "Naive Bayes Bagging")
    evaluate_model(boost, X_train, X_test, y_train, y_test, "Naive Bayes Boosting")

def run_neural_network(layers, name_suffix):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=44 + len(layers))
    base = MLPClassifier(hidden_layer_sizes=layers, max_iter=200, random_state=44)
    bag = BaggingClassifier(estimator=base, n_estimators=10, random_state=44, n_jobs=-1)

    evaluate_model(base, X_train, X_test, y_train, y_test, f"Neural Network {name_suffix}")
    evaluate_model(bag, X_train, X_test, y_train, y_test, f"Neural Network {name_suffix} Bagging")
    print(f"Neural Network {name_suffix} Boosting not supported")

def run_svm():
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=50)
    base = SVC(kernel='rbf', probability=True, random_state=50)
    bag = BaggingClassifier(estimator=base, n_estimators=5, random_state=50, n_jobs=-1)

    evaluate_model(base, X_train, X_test, y_train, y_test, "Support Vector Machine")
    evaluate_model(bag, X_train, X_test, y_train, y_test, "Support Vector Machine Bagging")
    print(f"Skipped because SVM with Boosting takes too much time")

# --- Run All ---
run_decision_tree('entropy')    # Gain Ratio
run_decision_tree('gini')       
run_decision_tree_cv('gini')
run_naive_bayes()
run_neural_network((10,), "1 hidden layer")
run_neural_network((10, 10), "2 hidden layers")
run_svm()
