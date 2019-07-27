#!/usr/bin/env python3
import csv
import os

import matplotlib.pyplot as plt


def main():
    datafiles = os.listdir("results/data")

    for datafile in datafiles:
        with open(f"results/data/{datafile}") as f:
            filename_info = datafile.rstrip(".csv").split("|")
            fairness_metric = filename_info[0]
            dataset_name = filename_info[1]

            data = [[float(x) for x in row] for row in (csv.reader(f, ))]

            # For now, assuming there are only FN, FP, and a fairness metric
            fn = [d[0] for d in data]
            fp = [d[1] for d in data]
            fairness = [min(2, d[2]) for d in data]
            test_acc = [d[3] for d in data]
            train_acc = [1 - d[0] - d[1] for d in data]

            summary = {
                # The accuracy of the model that performed the best on the test set
                "accuracy_of_best_model": min(data, key=lambda x: x[0] + x[1])[-1],
                "best_test_accuracy": max(test_acc),
                "best_train_accuracy": max(train_acc),
            }
            print(summary)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f'FNR vs FPR vs {fairness_metric}, {dataset_name} dataset')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('False Negative Rate')
            scatter = ax.scatter(fn, fp, c=fairness, cmap="viridis")
            fig.colorbar(scatter, ax=ax, label=f"{fairness_metric} (Capped at 2)")

            plt.savefig(f"results/figures/{os.path.basename(datafile).rstrip('.csv')}.png")

            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.set_title('Train Set vs Test Set Accuracy')
            # ax.set_xlabel('Train accuracy')
            # ax.set_ylabel('Test accuracy')
            #
            # scatter = ax.scatter(x=train_acc, y=test_acc, c=fairness, cmap="viridis")
            # fig.colorbar(scatter, ax=ax, label="Disparate Impact (Capped at 3)")
            #


if __name__ == '__main__':
    main()
