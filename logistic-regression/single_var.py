import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self) -> None:
        pass
    
    def fit(self, x_train, y_train):
        pass
    
    def predict(self, x_test):
        pass


if __name__ == "__main__":
    filepath = "../data/framingham.csv"
    df = pd.read_csv(filepath)
    
    print(df.head(10))


