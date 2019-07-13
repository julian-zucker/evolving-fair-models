#!/usr/bin/env python3
# Dataset is from https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29,
# and expected to be in data/german.data-numeric
import csv
import re
import requests


def main():
    with open("data/in/german.data-numeric", "w") as data_writing_fd:
        data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric")
        content = str(data.content, "utf8")
        # It uses multiple spaces to tabulate the data, we want commas
        content_with_comma_delimiter = "\n".join(
            [re.sub(" +", ",", line.strip()) for line in content.split("\n")]
        )
        data_writing_fd.write(content_with_comma_delimiter)

    with open("data/in/german.data-numeric") as data_reading_fd:
        data = list(csv.reader(data_reading_fd))

        features = [row[:-1] for row in data]
        with open("data/out/german.data", "w") as feature_file:
            data_writer = csv.writer(feature_file, csv.QUOTE_NONE)
            data_writer.writerows(features)

        labels = [row[-1] for row in data]
        with open("data/out/german.labels", "w") as label_file:
            label_writer = csv.writer(label_file, csv.QUOTE_NONE)
            label_writer.writerows(labels)


if __name__ == "__main__":
    main()
