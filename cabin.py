import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pdb

def build_ticket_df(df):

    ticket = df['Ticket'].str.split(' ', expand=True)

    ticket1 = ticket[0].to_list()
    ticket2 = ticket[1].to_list()
    ticket3 = ticket[2].to_list()

    ticket_num = []
    
    for t1, t2, t3 in zip(ticket1, ticket2, ticket3):
        try:
            ticket_num.append(int(t1))
        except:
            try:
                ticket_num.append(int(t2))
            except:
                try:
                    ticket_num.append(int(t3))
                except:
                    ticket_num.append(0)

    return pd.DataFrame(ticket_num)


def build_cabin_df(df):
    
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
            pass

    return pd.DataFrame(cabin)
    

if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    pd.set_option('display.max_rows', 10000)

    # 年齢の欠損値を補完
    a1 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'male')].fillna(40)
    a2 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'male')].fillna(30)
    a3 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'male')].fillna(25)
    a4 = df['Age'][(df['Pclass'] == 1) & (df['Sex'] == 'female')].fillna(35)
    a5 = df['Age'][(df['Pclass'] == 2) & (df['Sex'] == 'female')].fillna(28)
    a6 = df['Age'][(df['Pclass'] == 3) & (df['Sex'] == 'female')].fillna(21.5)
    df['Age'] = np.sum(pd.DataFrame([a1, a2, a3, a4, a5, a6]).fillna(0))

    # 旅客運賃の欠損値を補完
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # チケット番号を整理
    df['Ticket'] = build_ticket_df(df)

    # 客室番号の欠損値を補完
    df['Cabin'] = build_cabin_df(df)

    # 数値に変換
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)
    
    # 不要なデータを破棄
    df = df.drop(['PassengerId', 'Survived', 'Name'], axis =1)

    # 入出力データを生成
    X = df.drop('Cabin', axis=1)
    Y = df['Cabin']
    
    # 0〜1の範囲で正規化
    X = (X - X.min()) / (X.max() - X.min())

    
    pdb.set_trace()
    
