import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings

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
            cabin[i] = float(1.0)
        elif 'B' in str(c):
            cabin[i] = float(2.0)
        elif 'C' in str(c):
            cabin[i] = float(3.0)
        elif 'D' in str(c):
            cabin[i] = float(4.0)
        elif 'E' in str(c):
            cabin[i] = float(5.0)
        elif 'F' in str(c):
            cabin[i] = float(6.0)
        elif 'G' in str(c):
            cabin[i] = float(7.0)
        elif 'T' in str(c):
            cabin[i] = float(8.0)
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
    Z = df['Survived']
    df = df.drop(['PassengerId', 'Survived', 'Name'], axis=1)

    # Cabin が NaN を削除
    cabin_df = df.dropna(how='any')
    
    # 入出力データを生成
    X = cabin_df.drop('Cabin', axis=1)
    Y = cabin_df['Cabin']

    # 0〜1の範囲で正規化
    X = (X - X.min()) / (X.max() - X.min())
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    
    # モデルの学習
    GBDT = GradientBoostingRegressor()
    GBDT.fit(X, Y)
    
    # 回帰
    cabin_x = df[df.isnull().any(1)].drop(['Cabin'], axis=1)
    pred = GBDT.predict(cabin_x)

    j = 0
    for i, c in enumerate(df['Cabin']):
        if np.isnan(c):
            df['Cabin'][i] = pred[j]
            j += 1

    # 入出力データを生成
    X = df
    Y = Z
    
    # クロスバリデーション用のオブジェクトをインスタンス化する
    kfold_cv = KFold(n_splits=5, shuffle=True)
    warnings.filterwarnings('ignore')

    # classifier のアルゴリズムをすべて取得する
    all_Algorithms = all_estimators(type_filter="classifier")
    warnings.filterwarnings('ignore')
    
    max_clf = None
    max_score = -1
    
    # 各分類アルゴリズムをクロスバリデーションで評価する
    for (name, algorithm) in all_Algorithms:
        try:
            if (name == "LinearSVC"):
                clf = algorithm(max_iter = 10000)
            else:
                clf = algorithm()
            
            if hasattr(clf, "score"):
                scores = cross_val_score(clf, X, Y, cv=kfold_cv)
                print(name, "の正解率：")
                print(scores)
                if max_score < np.mean(scores):
                    max_clf = clf
                    max_score = np.mean(scores)
        except Exception as e:
            pass
        
    print(max_clf, max_score)
        
    # 平均正解率が最高だったモデルをトレーニング
    max_clf = max_clf.fit(X, Y)
    
    df_test = pd.read_csv('test.csv')
    passsengerid = df_test['PassengerId']

    # 年齢の欠損値を補完
    a1 = df_test['Age'][(df_test['Pclass'] == 1) & (df_test['Sex'] == 'male')].fillna(40)
    a2 = df_test['Age'][(df_test['Pclass'] == 2) & (df_test['Sex'] == 'male')].fillna(30)
    a3 = df_test['Age'][(df_test['Pclass'] == 3) & (df_test['Sex'] == 'male')].fillna(25)
    a4 = df_test['Age'][(df_test['Pclass'] == 1) & (df_test['Sex'] == 'female')].fillna(35)
    a5 = df_test['Age'][(df_test['Pclass'] == 2) & (df_test['Sex'] == 'female')].fillna(28)
    a6 = df_test['Age'][(df_test['Pclass'] == 3) & (df_test['Sex'] == 'female')].fillna(21.5)
    df_test['Age'] = np.sum(pd.DataFrame([a1, a2, a3, a4, a5, a6]).fillna(0))

    # 旅客運賃の欠損値を補完    
    df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

    # 客室番号の欠損値を補完
    cabin = df_test['Cabin'].to_list()
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
    df_test['Cabin'] = pd.DataFrame(cabin)

    # 数値に変換
    df_test['Sex'] = df_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    df_test['Embarked'] = df_test['Embarked'].fillna('S')
    df_test['Embarked'] = df_test['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)
    
    # 不要なデータを破棄
    df_test = df_test.drop(['PassengerId', 'Name', 'Ticket'], axis =1)

    # 0〜1の範囲で正規化
    X_test = (df_test - df_test.min()) / (df_test.max() - df_test.min())

    # 結果を出力
    pred = max_clf.predict(X_test)
    result = [int(i) for i in pred]

    submission = pd.DataFrame({'PassengerId':passsengerid, 'Survived':result})
    submission.to_csv('submission.csv' , index=False)

    
    
