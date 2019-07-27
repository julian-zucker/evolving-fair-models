#!/usr/bin/env python3
import csv
import math

import matplotlib.pyplot as plt
from matplotlib import colors


def main():
    with open("results/data/disparateimpact.csv") as f:
        # with open(sys.argv[1]) as f:
        data = [[float(x) for x in row] for row in (csv.reader(f, ))]
        summary = {
            # The accuracy of the model that performed the best on the test set
            "accuracy_of_best_model": min(data, key=lambda x: x[0] + x[1])[-1]
        }
        print(summary)

        # For now, assuming there are only FN, FP, and a fairness metric
        fn = [d[0] for d in data]
        fp = [d[1] for d in data]
        fairness = [min(3,d[2]) for d in data]
        test_set_acc = [d[3] for d in data]

        fig, ax = plt.subplots()
        ax.set_title('FNR vs FPR vs Disparate Impact, COMPAS dataset')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('False Negative Rate')
        # colors =
        print(min(fairness))
        print(max(fairness))

        scatter = ax.scatter(fn, fp, c=fairness, cmap="viridis",)
                             # norm=colors.LogNorm(vmin=min(fairness), vmax=max(fairness)),)
        fig.colorbar(scatter, ax=ax, label="Disparate Impact (Capped at 3)")
        plt.show()


if __name__ == '__main__':
    main()
