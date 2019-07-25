#!/usr/bin/env python3
"""This script reads all the datafiles from `./raw/`, parses them into a consistent data format,
and writes the results to `./test/` and `./train/`.

Invoke with `./data/script/generate_datafiles.py` from the top level."""

import csv
# The idea is that we can write each data-file processing script separately, and import
# and call them here to make re-generation easy.
import os
import random


def main():
    os.makedirs("data/raw/", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    random.seed(87333)

    ###############################################################################################
    # German dataset
    with open("data/raw/german.csv") as data_reading_fd:
        data = list(csv.reader(data_reading_fd))
        random.shuffle(data)

        features = [row[:-1] for row in data]
        labels = [row[-1] for row in data]
        # Ninth column is gender + marital status, 1,3,4 correspond to maleness
        gender = [row[8] in ["1", "3", "4"] for row in data]

        write_data("german", features, labels, gender)

    ###############################################################################################
    # COMPAS dataset
    with open("data/raw/compas.csv") as data_reading_fd:
        header = data_reading_fd.readline()
        data = list(csv.reader(data_reading_fd))
        random.shuffle(data)

        features = [row[:-1] for row in data]
        labels = [1 - float(row[-1]) for row in data]  # Invert 1 and 0. 0 here is a positive outcome
        priv = [row[3] == 1.0 for row in data]  # If race is 1.0 (Caucasian), privileged=True

        write_data("compas", features, labels, priv)


def write_data(filename, features, labels, priv):
    train_proportion = .8
    train_cutoff = int(len(features) * train_proportion)

    train_features = features[0:train_cutoff]
    test_features = features[train_cutoff:]

    labels = [[x] for x in labels]
    train_labels = labels[0:train_cutoff]
    test_labels = labels[train_cutoff:]

    # Make into 2D array for CSV reader
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


if __name__ == '__main__':
    main()
