import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings

import pdb

def extract_title(df):

    Title_Dictionary = {
        "Capt": 0, "Col": 0, "Major": 0, "Dr": 0, "Rev": 0,
        "Jonkheer": 1, "Don": 1, "Sir" : 1, "the Countess": 1, "Lady" : 1,
        "Mme": 2, "Ms": 2, "Mrs" : 2, 
        "Mlle": 3, "Miss" : 3,
        "Mr" : 4,
        "Master" : 5, "": 5,
    }
    return df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip()).map(Title_Dictionary)


def complement_age(df):

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


def clean_cabin(df):
    
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
            cabin[i] = float(0.0)

    return pd.DataFrame(cabin)


def process_df(df):

    # 名前から肩書きを抜き出す
    df['Title'] = extract_title(df)

    # 年齢の欠損値を補完する
    df['Age'] = complement_age(df)
    
    # 性別を数値に変換する
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    # 旅客運賃の欠損値を補完する
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 乗船港の欠損値を補完する
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)
    
    # チケット番号を整理する
    df['Ticket'] = build_ticket_df(df)

    # 客室番号を整理する
    df['Cabin'] = clean_cabin(df)
    
    return df


if __name__ == '__main__':

    # 訓練データを読み込み
    df = process_df(pd.read_csv('train.csv'))
    
    # 不要なデータを破棄
    Survived = df['Survived']
    df = df.drop(['PassengerId', 'Survived', 'Name'], axis=1)
    
    # 0〜1の範囲で正規化
    df = (df - df.min()) / (df.max() - df.min())
    
    # 入出力データを生成
    X = df
    Y = Survived
    
    # クロスバリデーション用のオブジェクトをインスタンス化する
    kfold_cv = KFold(n_splits=5, shuffle=False)
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
    
    # テストデータを読み込み
    df_test = process_df(pd.read_csv('test.csv'))
    passsengerid = df_test['PassengerId']
    
    # 不要なデータを破棄
    df_test = df_test.drop(['PassengerId', 'Name'], axis=1)
    
    # 0〜1の範囲で正規化
    X_test = (df_test - df_test.min()) / (df_test.max() - df_test.min())
    X_test = X_test.fillna(0)
    
    # 結果を出力
    pred = max_clf.predict(X_test)
    result = [int(i) for i in pred]
    
    submission = pd.DataFrame({'PassengerId':passsengerid, 'Survived':result})
    submission.to_csv('submission.csv', index=False)
    
