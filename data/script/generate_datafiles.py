#!/usr/bin/env python3
"""This script reads all the datafiles from `./raw/`, parses them into a consistent data format,
and writes the results to `./test/` and `./train/`.

Invoke with `./data/script/generate_datafiles.py` from the top level."""

import csv
import os
import random

import numpy as np
from sklearn.preprocessing import LabelEncoder


def main():
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    random.seed(87333)

    ###############################################################################################
    # German dataset
    with open("data/raw/german.csv") as data_reading_fd:
        data = list(csv.reader(data_reading_fd))
        random.shuffle(data)

        features = [row[:-1] for row in data]
        labels = [bool_to_outcome(row[-1] == "1") for row in data]
        # Ninth column is gender + marital status, 1,3,4 correspond to maleness
        gender = [row[8] in ["1", "3", "4"] for row in data]
        write_data("german", features, labels, gender)

    ###############################################################################################
    # COMPAS dataset
    with open("data/raw/compas.csv") as data_reading_fd:
        _ = data_reading_fd.readline()  # Skip header
        data = list(csv.reader(data_reading_fd))
        random.shuffle(data)

        features = [row[:-1] for row in data]
        # Remove one-hot encoding from crime stats, replace with index
        # Not the prettiest data representation, but induces better trees.
        features = [row[0:11] + [row[11:].index("1.0")] for row in features]

        labels = [bool_to_outcome(row[-1] == "1.0") for row in data]
        priv = [row[3] == "1.0" for row in data]  # If race is 1.0 (Caucasian), privileged=True

        write_data("compas", features, labels, priv)

    ###############################################################################################
    # Adult income dataset
    with open("data/raw/adult_income.csv") as data_reading_fd:
        _ = data_reading_fd.readline()  # Skip header
        data = [[item.strip() for item in row]
                for row in csv.reader(data_reading_fd, delimiter=",")]
        random.shuffle(data)

        t = np.transpose(data)
        result = []
        gender_encoder = None
        priv_encoder = None
        for col in t:
            try:
                result.append([int(x) for x in col])
            except:
                le = LabelEncoder().fit(col)

                if col[0] in ["Male", "Female"]:
                    gender_encoder = le

                if col[0] in ["<=50K", ">50K"]:
                    priv_encoder = le

                result.append(le.transform(col))

        result = np.transpose(result)

        labels = [bool_to_outcome(row[-1] == priv_encoder.transform([">50K"])[0])
                  for row in result]
        priv = [row[9] == gender_encoder.transform(["Male"])[0] for row in result]
        result = np.delete(result, -1, 1)

        write_data("adult_income", result, labels, priv)


def write_data(filename, features, labels, priv):
    train_proportion = .7
    train_cutoff = int(len(features) * train_proportion)

    train_features = features[0:train_cutoff]
    test_features = features[train_cutoff:]

    # Make into 2D array for CSV reader
    labels = [[x] for x in labels]
    train_labels = labels[0:train_cutoff]
    test_labels = labels[train_cutoff:]

    priv = [[x] for x in priv]
    train_privileged = priv[0:train_cutoff]
    test_privileged = priv[train_cutoff:]

    with open(f"data/train/{filename}.data", "w") as train_feature_file:
        csv.writer(train_feature_file, csv.QUOTE_NONE).writerows(train_features)

    with open(f"data/test/{filename}.data", "w") as test_feature_file:
        csv.writer(test_feature_file, csv.QUOTE_NONE).writerows(test_features)

    with open(f"data/train/{filename}.labels", "w") as train_label_file:
        csv.writer(train_label_file, csv.QUOTE_NONE).writerows(train_labels)

    with open(f"data/test/{filename}.labels", "w") as test_label_file:
        csv.writer(test_label_file, csv.QUOTE_NONE).writerows(test_labels)

    with open(f"data/train/{filename}.priv", "w") as train_privileged_file:
        csv.writer(train_privileged_file, csv.QUOTE_NONE).writerows(train_privileged)

    with open(f"data/test/{filename}.priv", "w") as test_privileged_file:
        csv.writer(test_privileged_file, csv.QUOTE_NONE).writerows(test_privileged)


def bool_to_outcome(b):
    return "Positive" if b else "Negative"


if __name__ == '__main__':
    main()
