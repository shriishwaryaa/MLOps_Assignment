import dvc.api
import pandas
import joblib
import json
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def getData():
    with dvc.api.open(repo="https://github.com/shriishwaryaa/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
        df = pandas.read_csv(fd)
    return df

def splitData(df):
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    train = X_train.copy()
    train["Class"] = y_train
    train.to_csv('../data/processed/train.csv')

    test = X_test.copy()
    test["Class"] = y_test
    test.to_csv('../data/processed/test.csv')

def trainModel():
    train = pandas.read_csv('../data/processed/train.csv')
    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]]

    model = RandomForestClassifier(n_estimators = 100, random_state=1, criterion="entropy")
    model.fit(X_train, y_train)
    return model

def saveModel(model):
    name = "../models/model.pkl"
    joblib.dump(model, name)

def testModel():
    test = pandas.read_csv('../data/processed/test.csv')
    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]]

    model = joblib.load('../models/model.pkl')
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    weightedf1score = f1_score(y_test, y_pred, average='weighted')

    write = {"Accuracy" : accuracy, "Weighted F1 Score" : weightedf1score}
    json_object = json.dumps(write, indent = 5)
    with open("../metrics/acc_f1.json", "w") as f:
        f.write(json_object)

def main():
    df = getData()
    splitData(df)
    model = trainModel()
    saveModel(model)
    testModel()

if __name__ == "__main__":
    main()
