"""Per-generation fitness CSV logger (for plotting/analysis)."""
import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, generation, best, mean):
        exists = self.path.exists()
        with self.path.open('a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['generation', 'best', 'mean'])
            w.writerow([generation, best, mean])
