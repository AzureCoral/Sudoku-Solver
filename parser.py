import numpy as np
import csv

class Parser():
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.grid = self.parse_csv()

    def parse_csv(self):
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            grid = []
            for row in reader:
                grid.append(row)
        return np.array(grid, dtype=int)

    def get_grid(self):
        return self.grid
