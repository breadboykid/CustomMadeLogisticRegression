import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression


if __name__ == "__main__":
    df = pd.read_csv("heart.csv")

    data = df.to_numpy()

    X = data[:, :-1]
    y = data[:, -1:]

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(X_train, y_train)
    
    model.predict(X_test)