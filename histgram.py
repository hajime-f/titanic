import pandas as pd
import matplotlib.pyplot as plt

import pdb

if __name__ == '__main__':

    df = pd.read_csv('train.csv')

    age1 = df.query('Pclass == 1').query('Sex == "male"')['Age']
    age2 = df.query('Pclass == 2').query('Sex == "male"')['Age']
    age3 = df.query('Pclass == 3').query('Sex == "male"')['Age']
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist([age1, age2, age3], bins=10)
    ax.set_xlabel('Age')

    plt.show()
    
    print(age1.median())
    print(age2.median())
    print(age3.median())

    age4 = df.query('Pclass == 1').query('Sex == "female"')['Age']
    age5 = df.query('Pclass == 2').query('Sex == "female"')['Age']
    age6 = df.query('Pclass == 3').query('Sex == "female"')['Age']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist([age4, age5, age6], bins=10)
    ax.set_xlabel('Age')
    
    print(age4.median())
    print(age5.median())
    print(age6.median())
    
    
    
