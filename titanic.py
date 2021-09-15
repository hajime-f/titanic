import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings

import pdb

if __name__ == '__main__':
    
    df = pd.read_csv('train.csv')

    # 欠損値を補完
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 数値に変換
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    # 不要なデータを破棄
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis =1)

    # 0〜1の範囲で正規化
    df = (df - df.min()) / (df.max() - df.min())

    # 入出力データを生成
    X = df.drop('Survived', axis=1)
    Y = df.Survived
    
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
        
    # 平均正解率が最高だったモデルをトレーニング
    max_clf = max_clf.fit(X, Y)
    
    df_test = pd.read_csv('test.csv')
    passsengerid = df_test['PassengerId']
    
    # 欠損値を補完
    df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
    df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())

    # 数値に変換
    df_test['Sex'] = df_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    # 不要なデータを破棄
    df_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis =1)

    # 0〜1の範囲で正規化
    X_test = (df_test - df_test.min()) / (df_test.max() - df_test.min())

    # 結果を出力
    pred = max_clf.predict(X_test)
    result = [int(i) for i in pred]

    submission = pd.DataFrame({'PassengerId':passsengerid, 'Survived':result})
    submission.to_csv('submission.csv' , index=False)

