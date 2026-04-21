"""Reads a fitness CSV (generation,best,mean) and plots the evolution curve."""
import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("usage: plot_fitness.py <file.csv>")
        sys.exit(1)
    path = Path(sys.argv[1])
    with path.open() as f:
        list(csv.DictReader(f))
    raise NotImplementedError("plot with matplotlib")


if __name__ == '__main__':
    main()
