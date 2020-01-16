# Renders the csv sin two_fairness_metrics
import csv
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def main():
    dir = "results/data/two_fairness_metrics"
    datafiles = os.listdir(dir)

    for datafile in datafiles:
        with open(f"{dir}/{datafile}") as f:
            filename_info = datafile.rstrip(".csv").split("|")
            fairness_metric1 = filename_info[0]
            fairness_metric2 = filename_info[1]
            dataset_name = filename_info[2]

            for _ in range(3):
                f.readline()
            header = f.readline()
            if filename_info.index(fairness_metric1) < filename_info.index(fairness_metric2):
                f1index, f2index = (3, 2)
            else:
                f1index, f2index = (2, 3)

            data = [[float(x) for x in row] for row in (csv.reader(f))]

            fn = [d[0] for d in data]
            fp = [d[1] for d in data]
            fairness1 = [min(3, d[f1index]) for d in data]
            fairness2 = [min(3, d[f2index]) for d in data]
            test_acc = [d[4] for d in data]
            train_acc = [1 - d[0] - d[1] for d in data]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f'{fairness_metric1}, {fairness_metric2}, and Accuracy on {dataset_name}')
        ax.set_xlabel(f'{fairness_metric1}')
        ax.set_ylabel(f'{fairness_metric2}')
        scatter = ax.scatter(fairness1, fairness2, c=test_acc, s=7, cmap="viridis_r")
        fig.colorbar(scatter, ax=ax, label="Accuracy")

        plt.savefig(f"results/figures/two_metrics/{os.path.basename(datafile).rstrip('.csv')}.png",
                    bbox_inches='tight')


if __name__ == '__main__':
    main()
