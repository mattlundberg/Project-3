import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data_importer = DataImporter('resources/test.tsv')
    df = data_importer.import_data_tsv()
    print(df.head())

class DataImporter:
    def __init__(self, path: str):
        self.path = path

    def import_data_tsv(self):
        df = pd.read_csv(self.path, delimiter='\t')
        return df

    def import_data_csv(self):
        df = pd.read_csv(self.path)
        return df

if __name__ == "__main__":
    main()
