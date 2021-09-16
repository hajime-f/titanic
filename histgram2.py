import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pdb

if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    
    cabin = df['Cabin'].to_list()
    for i, c in enumerate(cabin):
        if 'A' in str(c):
            cabin[i] = 1
        elif 'B' in str(c):
            cabin[i] = 2
        elif 'C' in str(c):
            cabin[i] = 3
        elif 'D' in str(c):
            cabin[i] = 4
        elif 'E' in str(c):
            cabin[i] = 5
        elif 'F' in str(c):
            cabin[i] = 6
        elif 'G' in str(c):
            cabin[i] = 7
        else:
            cabin[i] = 0

    hist_survived = [0] * 8
    hist_unsurvived = [0] * 8
    survived = df['Survived'].to_list()

    for i in range(len(cabin)):
        if survived[i]:
            hist_survived[cabin[i]] += 1
        else:
            hist_unsurvived[cabin[i]] += 1

    ratio = [0] * 8
    for i in range(len(ratio)):
        h = hist_survived[i] + hist_unsurvived[i]
        ratio[i] = hist_survived[i] / h

    print(ratio)
