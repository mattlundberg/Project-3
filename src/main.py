import pandas as pd
#from datasets import load_dataset

def main():
    data_importer = DataImporter()
    data_importer.import_data()
    print(data_importer.get_validation_data().head())

class DataImporter:
    def __init__(self):
        self.ds_train = pd.DataFrame()
        self.ds_test = pd.DataFrame()
        self.ds_validation = pd.DataFrame()
        self.dataset = None

    def import_data(self):
        self.dataset = load_dataset(path="ucsbnlp/liar",revision="main", trust_remote_code=True)
        self.ds_train = self.dataset['train'].to_pandas()
        self.ds_test = self.dataset['test'].to_pandas()
        self.ds_validation = self.dataset['validation'].to_pandas()

    def get_train_data(self):
        return self.ds_train

    def get_test_data(self):
        return self.ds_test

    def get_validation_data(self):
        return self.ds_validation



if __name__ == "__main__":
    main()
