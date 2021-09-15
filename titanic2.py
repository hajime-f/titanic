import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pdb

if __name__ == '__main__':
    
    df = pd.read_csv('train.csv')

    # 欠損値を補完
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # 数値に変換
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['Embarked'] = df['Embarked'].map( {'S': 0 , 'C':1 , 'Q':2}).astype(int)

    # 不要なデータを破棄
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis =1)

    # 0〜1の範囲で正規化
    df = (df - df.min()) / (df.max() - df.min())

    # 入出力データを生成
    train_X = df.drop('Survived', axis=1)
    train_y = df.Survived
    (train_X , test_X , train_y , test_y) = train_test_split(train_X, train_y , test_size = 0.3 , random_state = 0)
    
    clf = RandomForestClassifier(n_estimators = 10,max_depth=5,random_state = 0)
    clf = clf.fit(train_X , train_y)
    pred = clf.predict(test_X)
    print(accuracy_score(pred,test_y))

    fin = pd.read_csv('test.csv')
    fin.head()
    
    passsengerid = fin['PassengerId']
    fin.isnull().sum()
    fin['Fare'] = fin['Fare'].fillna(fin['Fare'].median())
    fin['Age'] = fin['Age'].fillna(fin['Age'].median())
    fin['Embarked'] = fin['Embarked'].fillna('S')
    
    #カテゴリ変数の変換
    fin['Sex'] = fin['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    fin['Embarked'] = fin['Embarked'].map( {'S': 0 , 'C':1 , 'Q':2}).astype(int)
    
    #不要なcolumnを削除
    fin= fin.drop(['Cabin','Name','Ticket','PassengerId'],axis =1)

    #ランダムフォレストで予測
    predictions = clf.predict(fin)
    
    submission = pd.DataFrame({'PassengerId':passsengerid, 'Survived':predictions})
    submission.to_csv('submission.csv' , index = False)
    
