from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_text
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data():
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    print("Metadata:")
    print(bank_marketing.metadata)
    print("\nVariables:")
    print(bank_marketing.variables)
    return X, y

def preprocess_data(X, y):
    X_encoded = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

def visualize_tree(clf, X):
    """Visualize the Decision Tree with enhanced readability."""
    plt.figure(figsize=(30,20))
    
    plot_tree(
        clf,
        filled=True,
        feature_names=X.columns,
        class_names=['Will not buy', 'Will buy'],
        rounded=True,
        proportion=True,
        fontsize=12,
        impurity=False,  
        max_depth=5  
    )
    
    plt.title("Decision Tree", fontsize=20)
    plt.show()
    

    tree_text = export_text(clf, feature_names=list(X.columns), max_depth=5)
    print(tree_text)
def main():
    X, y = fetch_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    clf = train_model(X_train, y_train)
    evaluate_model(clf, X_test, y_test)
    visualize_tree(clf, X_train)

if __name__ == "__main__":
    main()

