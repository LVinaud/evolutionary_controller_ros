"""Lê um CSV de fitness (geracao,melhor,media) e plota a curva de evolução."""
import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("uso: plotar_fitness.py <arquivo.csv>")
        sys.exit(1)
    caminho = Path(sys.argv[1])
    with caminho.open() as f:
        list(csv.DictReader(f))
    raise NotImplementedError("plotar com matplotlib")


if __name__ == '__main__':
    main()
