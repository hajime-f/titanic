import pandas as pd
import matplotlib.pyplot as plt

import pdb

if __name__ == '__main__':

    df = pd.read_csv('train.csv')

    plt.scatter(df['Survived'], df['Fare'])
    plt.show()

