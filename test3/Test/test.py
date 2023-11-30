# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 00:08:26 2022

@author: alsdk
"""

# 환경 준비
import pandas as pd
import numpy as np
from sklearn import tree, neighbors, svm, linear_model
from sklearn import model_selection
from sklearn import preprocessing
from tabulate import tabulate
import pickle
import joblib
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
# pd.set_option('display.max_columns', None)


# 전처리 (학습을 위해 object형을 벡터형으로 변환시켜준다.)
def prepro(data):
    
# =============================================================================
#       LabelEncoder를 거친 OneHotEncoding  (실패)
# =============================================================================
  
    # # 2차원 배열의 Dataframe 모든 원소를 1차원 배열로 만들기
    # data = data.values.reshape(-1)
    # # 레이블 인코딩 과정을 거쳐서 문자열에 숫자 하나씩 매칭
    # encoder = preprocessing.LabelEncoder()
    # encoder.fit(data)
    # x = encoder.transform(data)
    # # 인덱스 값들을 2차원 배열로 변환
    # x = x.reshape(-1,1)
    
    # encoder = preprocessing.OneHotEncoder(sparse=False)
    # onehot = encoder.fit(x)   # OneHot 생성자 만들고 학습시키기
    # x = encoder.transform(x)   # 학습 후 결과값 얻기
    
    
# =============================================================================
#     OneHotEncoder 클래스 사용
# =============================================================================
    
    # OneHotEncoder (넘파이 배열 반환 / 보이지 않는 범주 값 처리)
    ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
    cap_shape = ohe.fit_transform(data[['cap-shape']])
    print(ohe.categories_)
    # numpy.array를 Dataframe으로 변환
    print(pd.DataFrame(cap_shape, columns=['cap-shpae_'+col for col in ohe.categories_[0]]))
    x = pd.concat([data.drop(columns=['cap-shape']),pd.DataFrame(cap_shape,columns=['cap-shape_' + col for col in ohe.categories_[0]])], axis=1)
    joblib.dump(ohe, 't0.t')
    
    # 학습할 columns
    list_c = ['cap-color','odor','stalk-shape']
    list_a = ['cc', 'od', 'ss']
    
    j = 0
    for i in list_a:
        i = ohe.fit_transform(data[[list_c[j]]])
        string = list_c[j]+'_'
        x1 = pd.concat([x.drop(columns=[list_c[j]]), pd.DataFrame(i, columns=[string + col for col in ohe.categories_[0]])], axis=1)
        # print('\nx1: ',x1)
        x = x1
        j+=1
        joblib.dump(ohe, 't'+str(j)+'.t')
    print(x)
    
    
# =============================================================================
#     pandas를 사용하여 숫자형 변환없이 OneHot 처리
# =============================================================================
  
    # x = pd.get_dummies(data)
    # print(x)
    

    print(x.shape)
    return x


def get_data():
    data = pd.read_csv("../data/mushrooms.csv")
    # 결측치 확인
    print(data.info())
    # -> 현재 데이터의 값들이 문자열 형태이다 
    print(data.shape)
    

    # 데이터 분리
    x = data.iloc[:,1:]
    y = data['poisonous']
    
    # column 제거 (분류 기준이 많을 수록 정확도는 높지만 서비스 이용자가 일반인임을 간주하여 일반인의 눈으로 아래 모든 사항들을 조사하기는 어려우므로 최소의 정보들로 최대의 효율)
    list = [ 'cap-surface','bruises','gill-attachment','gill-spacing', 'gill-size','gill-color','stalk-root',
            'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 
            'veil-type', 'ring-number', 'ring-type','spore-print-color',
            'veil-color','habitat','population']
    for i in list:
        x = x.drop(i, axis=1)
    
    print(x.shape)
    print(y.shape)
    
    # Label의 분포도? 확인
    print(y.value_counts())
    
    # 문자열인 알파벳을 수치화 해줘야 한다(인코딩)
    x = prepro(x)
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=(11), test_size=0.3)

    return x_train, x_test, y_train, y_test

def make_model():
    model = tree.DecisionTreeClassifier()
    # model = svm.SVC()
    # model = neighbors.KNeighborsClassifier()
    # model = linear_model.LogisticRegression()
    return model

def do_learn():
  x_train, x_test, y_train, y_test = get_data()
  model = make_model()
  
  # cv = 교차 검증 횟수 지정
  model_result = model_selection.cross_val_score(model, x_train, y_train, cv=5)
  print("교차 검증 결과 : ", model_result)
  print("교차검증 결과 평균값 : ", model_result.mean())
  
  model.fit(x_train, y_train)
  score = model.score(x_test, y_test)
  print("점수 : ",score)
  pickle.dump(model,open('m1.m','wb'))
  
do_learn()
print('\n\n\n')
  

def load_data(cap_shape, cap_color, odor, stalk_shape):
    
    data = pd.DataFrame({'cap-shape':[cap_shape], 
                         'cap-color':[cap_color],
                         'odor' : [odor],
                         'stalk-shape' : [stalk_shape]})
    ohe0, ohe1, ohe2, ohe3 = load_encoder()
    ohe = [ohe0, ohe1, ohe2, ohe3]
   
    # 예측할 columns
    list_c = ['cap-shape','cap-color','odor','stalk-shape']
    list_a = ['cs','cc', 'od', 'ss']
    
    
    print(data)
    
    
    j = 0
    for i in list_a:
        i = ohe[j].transform(data[[list_c[j]]])
        string = list_c[j]+'_'
        x = pd.concat([data.drop(columns=[list_c[j]]), pd.DataFrame(i, columns=[string + col for col in ohe[j].categories_[0]])], axis=1)
        # print('\nx1: ',x1)
        data = x
        j+=1
    print(data)
       
    return data

def load_model():
    model = pickle.load(open("m1.m","rb"))
    return model
def load_encoder():
    encoder1 = joblib.load('t0.t')
    encoder2 = joblib.load('t1.t')
    encoder3 = joblib.load('t2.t')
    encoder4 = joblib.load('t3.t')
    return encoder1, encoder2, encoder3, encoder4

def do_predict(model, cap_shape, cap_color, odor, stalk_shape):
    
    x = load_data(cap_shape, cap_color, odor, stalk_shape)
    
    # model이 예측한 값
    y_pre = model.predict(x)
    print(y_pre)
    
    return y_pre

model = load_model()
do_predict(model, 'b', 'w', 'l', 'e')


