import csv
import sys
import os

def main(filename):
    dir = "results/data/one_fairness_metric/"
    for filename in os.listdir(dir):
        if "Disparate Impact" in filename:
            with open(dir + filename, 'r') as f:
                for _ in range(4):
                    f.readline()
                reader = csv.reader(f)
                best_acc = 0
                disparate_impact = None

                for row in reader:
                    if float(row[2]) > 1.8:
                        continue

                    acc = 1 - float(row[0]) - float(row[1])
                    if acc > best_acc:
                        best_acc = acc
                        disparate_impact = float(row[2])

            print(f"File: {filename} Acc: {best_acc}, DI: {disparate_impact}")


if __name__ == '__main__':
    main(sys.argv[1])
