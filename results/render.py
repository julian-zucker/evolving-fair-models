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
            fairness = [min(5, d[2]) for d in data]
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
            print(summary)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f'FNR, FPR, and {fairness_metric} on the {dataset_name} dataset')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('False Negative Rate')
            scatter = ax.scatter(fn, fp, c=fairness, cmap="viridis")
            fig.colorbar(scatter, ax=ax, label=f"{fairness_metric}")

            plt.savefig(f"results/figures/{os.path.basename(datafile).rstrip('.csv')}.png")

            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.set_title('Accuracy vs Fairness')
            # ax.set_xlabel(f'{fairness_metric}')
            # ax.set_ylabel('Accuracy')
            #
            # scatter = ax.scatter(x=fairness, y=test_acc)
            # plt.show()


if __name__ == '__main__':
    main()
