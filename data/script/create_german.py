#!/usr/bin/env python3
# Dataset is from https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29,
# and expected to be in data/german.data-numeric
import csv
import random
import re

import requests


def main():
    with open("data/raw/german.data-numeric", "w") as data_writing_fd:
        data = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric")
        content = str(data.content, "utf8")
        # It uses multiple spaces to tabulate the data, we want commas
        content_with_comma_delimiter = "\n".join(
            [re.sub(" +", ",", line.strip()) for line in content.split("\n")]
        )
        data_writing_fd.write(content_with_comma_delimiter)

    with open("data/raw/german.data-numeric") as data_reading_fd:
        data = list(csv.reader(data_reading_fd))
        random.seed(87333)
        random.shuffle(data)

        # Ninth column is gender + marital status, 1,3,4 correspond to maleness
        gender = [[row[8] in ["1", "3", "4"]] for row in data]

        train_proportion = .8
        train_cutoff = int(len(data) * train_proportion)

        train_features = [row[:-1] for row in data][0:train_cutoff]
        with open("data/train/german.data", "w") as train_feature_file:
            train_data_writer = csv.writer(train_feature_file, csv.QUOTE_NONE)
            train_data_writer.writerows(train_features)

        test_features = [row[:-1] for row in data][train_cutoff:]
        with open("data/test/german.data", "w") as test_feature_file:
            test_data_writer = csv.writer(test_feature_file, csv.QUOTE_NONE)
            test_data_writer.writerows(test_features)

        train_labels = [row[-1] for row in data][0:train_cutoff]
        with open("data/train/german.labels", "w") as train_label_file:
            train_label_writer = csv.writer(train_label_file, csv.QUOTE_NONE)
            train_label_writer.writerows(train_labels)

        test_labels = [row[-1] for row in data][train_cutoff:]
        with open("data/test/german.labels", "w") as test_label_file:
            test_label_writer = csv.writer(test_label_file, csv.QUOTE_NONE)
            test_label_writer.writerows(test_labels)

        train_privileged = gender[0:train_cutoff]
        with open("data/train/german.priv", "w") as train_privileged_file:
            train_privileged_writer = csv.writer(train_privileged_file, csv.QUOTE_NONE)
            train_privileged_writer.writerows(train_privileged)

        test_privileged = gender[train_cutoff:]
        with open("data/test/german.priv", "w") as test_privileged_file:
            test_privileged_writer = csv.writer(test_privileged_file, csv.QUOTE_NONE)
            test_privileged_writer.writerows(test_privileged)


if __name__ == "__main__":
    main()
