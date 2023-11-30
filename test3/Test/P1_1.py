# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:07:20 2022

@author: alsdk
"""

from flask import Flask, request, render_template
import pickle
import joblib
import pandas as pd

# def get_data():
# =============================================================================
# 0,  1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2
# e,  b, ,w, ,l, , , , ,e, , , , , , ,w, , , ,s,m
# p,  x, ,w, ,p, , , , ,e, , , , , , ,w, , , ,v,g
# =============================================================================
#     # 다중 배열로 
#     x = [[5.7,2.8,4.1,1.3],[5.8,2.7,5.1,1.9]]

#     return x

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

# items = [0,0,0,0]
# def connecting(item, i):
#     items[i] = item
    

model = load_model()

# =============================================================================
# =============================================================================

webserver = Flask(__name__)
print(__name__)


@webserver.route("/")
def index():
    msg = "Welcome!"
    return msg

@webserver.route("/mushroom")
def mushroom():
    msg = render_template("mushroom.html")
    return msg

@webserver.route("/mushroom/survey", methods=['GET', 'POST'])
def survey():
    msg = render_template("survey_1.html")
    return msg


# method가 POST인 형식만 받겠다.
@webserver.route("/mushroom/ans", methods=["POST"])
def mushroom_ans():
    # val = request.form.get('ss')
    # connecting(val, 3)
    
    # cap_shape = str(items[0])
    # cap_color = str(items[1])
    # odor = str(items[2])
    # stalk_shape = str(items[3])
    cap_shape = request.values.get('cs')
    cap_color = request.values.get('color')
    odor = request.values.get('od')
    stalk_shape = request.values.get('ss')
    
    y_pre = do_predict(model, cap_shape, cap_color, odor, stalk_shape)
    # # format String
    # msg = f"{cap_shape} {cap_color} {odor} {stalk_shape}=> {y_pre[0]}"
    
    pred=y_pre[0]
    if pred == "e":
        fin = render_template("result.html")
        return fin
    elif pred == "p":
        fin = render_template("result1.html")
        return fin
        
    #fin = render_template("result1.html", cs=cap_shape,cc=cap_color,od=odor,ss=stalk_shape,pred=y_pre[0])
    
    

# =============================================================================
# #Go to next page
# @webserver.route('/mushroom/<nxtPage>')
# def click(nxtPage):
#     return render_template(nxtPage+".html")
# =============================================================================

webserver.run(port=2022, debug=False)