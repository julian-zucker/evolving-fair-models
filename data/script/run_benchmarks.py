import csv

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


def main():
    np.random.seed(8799)
    for filename in ["german", "COMPAS", "adult_income", "taiwan"]:
        run_benchmark(filename)


def run_benchmark(filename):
    print("=====================")
    print(filename)
    with open(f"data/train/{filename}.data", "r") as train_feature_file:
        train_features = list(csv.reader(train_feature_file))

    with open(f"data/train/{filename}.labels", "r") as train_label_file:
        train_labels = [prediction.strip() == "Positive" for prediction in
                        train_label_file.readlines()]

    model = RandomizedSearchCV(DecisionTreeClassifier(),
                               {
                                   "min_samples_split": np.arange(2, 10),
                                   "min_samples_leaf": np.arange(2, 10),
                                   "max_depth": np.arange(1, 10),
                                   "min_impurity_decrease": np.arange(0.0, 0.5)
                               }, cv=3)
    model.fit(train_features, train_labels)

    with open(f"data/test/{filename}.data", "r") as test_feature_file:
        test_features = list(csv.reader(test_feature_file))

    with open(f"data/test/{filename}.labels", "r") as test_label_file:
        test_labels = [prediction.strip() == "Positive" for prediction in
                       test_label_file.readlines()]

    with open(f"data/test/{filename}.priv", "r") as test_priv_file:
        test_priv = [priv.strip() == "True" for priv in test_priv_file]

    predictions = model.predict(test_features)
    accuracy = np.mean(np.equal(predictions, test_labels))

    priv = []
    priv_labels = []
    nonpriv = []
    nonpriv_labels = []
    for datapoint, label, datapoint_priv in zip(test_features, test_labels, test_priv):
        if datapoint_priv:
            priv.append(datapoint)
            priv_labels.append(label)
        else:
            nonpriv.append(datapoint)
            nonpriv_labels.append(label)

    priv_pos = np.mean(model.predict(priv))
    non_priv_pos = np.mean(model.predict(nonpriv))

    print(priv_pos, non_priv_pos)

    print(f"Accuracy: {accuracy}")
    print(f"Disparate impact: {max(priv_pos / non_priv_pos, non_priv_pos / priv_pos)}")


if __name__ == '__main__':
    main()
