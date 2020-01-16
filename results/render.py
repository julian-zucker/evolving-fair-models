#!/usr/bin/env python3
import csv
import os

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f'FNR, FPR, and {fairness_metric} on the {dataset_name} dataset')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('False Negative Rate')
            scatter = ax.scatter(fn, fp, c=fairness, cmap="viridis_r")
            fig.colorbar(scatter, ax=ax, label=f"{fairness_metric}")

            plt.savefig(
                f"results/figures/one_metric/{os.path.basename(datafile).rstrip('.csv')}.png",
                bbox_inches='tight'
            )


if __name__ == '__main__':
    main()
