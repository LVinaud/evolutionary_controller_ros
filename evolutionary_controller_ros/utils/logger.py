"""Logger CSV de fitness por geração (para plot/análise)."""
import csv
from pathlib import Path


class LoggerCSV:
    def __init__(self, caminho):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)

    def registrar(self, geracao, melhor, media):
        existe = self.caminho.exists()
        with self.caminho.open('a', newline='') as f:
            w = csv.writer(f)
            if not existe:
                w.writerow(['geracao', 'melhor', 'media'])
            w.writerow([geracao, melhor, media])
