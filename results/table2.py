#!/usr/bin/env python3
import csv
import json
import os

def main():
    dir = "results/data/one_fairness_metric"
    datafiles = os.listdir(dir)

    for datafile in datafiles:
        with open(f"{dir}/{datafile}") as f:
            filename_info = datafile.rstrip(".csv").split("|")
            fairness_metric = filename_info[0]
            dataset_name = filename_info[1]

            data = [[float(x) for x in row] for row in (csv.reader(f, ))]

            fn = [d[0] for d in data]
            fp = [d[1] for d in data]
            fairness = [min(3, d[2]) for d in data]
            test_acc = [d[3] for d in data]
            train_acc = [1 - d[0] - d[1] for d in data]





            summary = {
                # The accuracy of the model that performed the best on the test set
                "dataset": dataset_name,
                "fairness_metric": fairness_metric,
                "accuracy_of_best_model": min(data, key=lambda x: x[0] + x[1])[-1],
                "best_test_accuracy": max(test_acc),
                "best_train_accuracy": max(train_acc),
            }

            for fairness_cutoff in [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3]:
                best_acc = 0
                for item in data:
                    if item[2] <= fairness_cutoff:
                        best_acc = max(best_acc, (1 - item[0] - item[1]))
                summary[f"best_accuracy_below_fairness_{fairness_cutoff}"] = best_acc

            print(json.dumps(summary, indent="\t"))

if __name__ == '__main__':
    main()
