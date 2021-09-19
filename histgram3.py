import pandas as pd
import matplotlib.pyplot as plt

import pdb

def extract_title():

    Title_Dictionary = {
        "Capt": 0,
        "Col": 0,
        "Major": 0,
        "Jonkheer": 1,
        "Don": 1,
        "Sir" : 1,
        "Dr": 0,
        "Rev": 0,
        "the Countess": 1 ,
        "Mme": 2,
        "Mlle": 3,
        "Ms": 2,
        "Mr" : 4,
        "Mrs" : 2,
        "Miss" : 3,
        "Master" : 5,
        "Lady" : 1,
        "": 5
    }
    
    return df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip()).map(Title_Dictionary)


if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    pd.set_option('display.max_rows', 10000)
    
    df['Title'] = extract_title()
    
    df['Died'] = 1 - df['Survived']
    df.groupby('Title').aggregate('mean')[['Survived', 'Died']].plot(kind='bar', stacked='True')
    
    plt.show()
    
    


