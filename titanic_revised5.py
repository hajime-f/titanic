import numpy as np
import pandas as pd

import pdb

def extract_title():

    Title_Dictionary = {
        "Capt": 0, "Col": 0, "Major": 0, "Dr": 0, "Rev": 0,
        "Jonkheer": 1, "Don": 1, "Sir" : 1, "the Countess": 1, "Lady" : 1,
        "Mme": 2, "Ms": 2, "Mrs" : 2, 
        "Mlle": 3, "Miss" : 3,
        "Mr" : 4,
        "Master" : 5, "": 5,
    }
    return df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip()).map(Title_Dictionary)


def complement_age():

    # 年齢の欠損値を補完
    a1 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'male') & (df['Title'] == 0)].fillna(51.0)
    a2 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'male') & (df['Title'] == 1)].fillna(40.0)
    a3 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'male') & (df['Title'] == 2)].fillna(36.0)
    a4 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'male') & (df['Title'] == 3)].fillna(36.0)
    a5 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'male') & (df['Title'] == 4)].fillna(40.0)
    a6 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'male') & (df['Title'] == 5)].fillna(4.0)

    a7 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'male') & (df['Title'] == 0)].fillna(46.5)
    a8 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'male') & (df['Title'] == 1)].fillna(36.0)
    a9 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'male') & (df['Title'] == 2)].fillna(34.0)
    a10 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'male') & (df['Title'] == 3)].fillna(34.0)
    a11 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'male') & (df['Title'] == 4)].fillna(31.0)
    a12 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'male') & (df['Title'] == 5)].fillna(1.0)

    a13 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'male') & (df['Title'] == 0)].fillna(22.0)
    a14 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'male') & (df['Title'] == 1)].fillna(22.0)
    a15 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'male') & (df['Title'] == 2)].fillna(22.0)
    a16 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'male') & (df['Title'] == 3)].fillna(22.0)
    a17 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'male') & (df['Title'] == 4)].fillna(26.0)
    a18 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'male') & (df['Title'] == 5)].fillna(4.0)

    a19 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Title'] == 0)].fillna(49.0)
    a20 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Title'] == 1)].fillna(40.5)
    a21 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Title'] == 2)].fillna(40.0)
    a22 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Title'] == 3)].fillna(30.0)
    a23 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Title'] == 4)].fillna(35.0)
    a24 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Title'] == 5)].fillna(35.0)

    a25 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'female') & (df['Title'] == 0)].fillna(24.0)
    a26 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'female') & (df['Title'] == 1)].fillna(24.0)
    a27 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'female') & (df['Title'] == 2)].fillna(31.5)
    a28 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'female') & (df['Title'] == 3)].fillna(24.0)
    a29 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'female') & (df['Title'] == 4)].fillna(24.0)
    a30 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'female') & (df['Title'] == 5)].fillna(24.0)

    a31 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'female') & (df['Title'] == 0)].fillna(18.0)
    a32 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'female') & (df['Title'] == 1)].fillna(18.0)
    a33 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'female') & (df['Title'] == 2)].fillna(31.0)
    a34 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'female') & (df['Title'] == 3)].fillna(18.0)
    a35 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'female') & (df['Title'] == 4)].fillna(18.0)
    a36 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'female') & (df['Title'] == 5)].fillna(18.0)

    return np.sum(pd.DataFrame([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                a11, a12, a13, a14, a15, a16, a17, a18, a19, a20,
                                a21, a22, a23, a24, a25, a26, a27, a28, a29, a30,
                                a31, a32, a33, a34, a35, a36]).fillna(0))


if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    pd.set_option('display.max_rows', 10000)
    
    # 名前から肩書きを抜き出して「Title」に入れる
    df['Title'] = extract_title()

    # 肩書きを one-hot に展開する
    title_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, title_dummies], axis=1)

    # 年齢の欠損値を補完する
    df['Age'] = complement_age()
    
    # 性別を数値に変換する
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    # 旅客運賃の欠損値を補完
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 乗船港の欠損値を補完し、one-hot に展開する
    df['Embarked'] = df['Embarked'].fillna('S')
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    


    # 不要なデータを破棄
    Survived = df['Survived']
    df = df.drop(['PassengerId', 'Survived', 'Name', 'Title', 'Embarked'], axis=1)
    
    
