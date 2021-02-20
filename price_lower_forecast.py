import pandas as pd
import datetime
import odps as od
import sys
import get_data_from_odps as gdfo
import numpy as np
import datetime
from collections import defaultdict
from sklearn import tree    
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from odps import ODPS
from odps import options
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(): # loadatas
    data =gdfo.select_table_to_dl()
    data= data[data['price_unit'] == '20GP']
    return data

def pre_data():
    data = load_data()
    data =data[['linecodeparentid','linecodename','pol_name','pod_name','price','etd','vessel_name']]
    # to datetime
    data['etd'] =pd.to_datetime(data['etd']).dt.floor('d')
    data['etd_days']= (data['etd'] -pd.to_datetime('2020-01-01')).map(lambda x:x.days)
    data['etd_mouth'] = data['etd'].dt.month
    data['etd_q'] =(data['etd'].dt.to_period('Q')).apply(lambda x : str(x)[-2:]).str.replace('Q', '')
    return data 
def col_labels():
    dataset =pre_data()
    dataset =dataset[dataset['etd']<= datetime.datetime.now()]
    for col in ['linecodeparentid','linecodename','pol_name','pod_name','vessel_name']:
        gle =LabelEncoder()
        gle.fit(dataset[col])
        dataset[col] = gle.transform(dataset[col])
        dataset_price =pd.DataFrame(dataset.groupby(['pol_name','pod_name','vessel_name'],as_index=False)['price'].agg(percentile(50)))
        dataset_index = dataset_price.reset_index()
        maersk_price_dataset = pd.merge(dataset,dataset_index,on =['pol_name','pod_name','vessel_name'])
    return maersk_price_dataset
        
# 寻找区间
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def data_50_percentile():

    maersk_price_dataset =col_labels()
    x =maersk_price_dataset.drop(['price_x','etd','price_y','index'],axis =1)
    y =maersk_price_dataset['price_y']
    y = np.array(y, dtype=int)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state =1)
    return x_train,x_test,y_train,y_test

def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)


def prediction_to_odps(clf):
    odps =gdfo.connect_odps()
    x_train,x_test,y_train,y_test = data_50_percentile()
    model =clf.fit(x_train,y_train)

    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    mape = MAPE(y_test, result)
    print('mape: %f,score: %f'%(clf.score(x_test, y_test),MAPE(y_test, result)))
    data_plot =pd.DataFrame()
    ds =x_test[:30]
    ds['y_test'] =y_test[:30]
    ds['result'] =result[:30]
#     print(ds)
    data_plot['y_test'] =y_test[:100]
    data_plot = data_plot.reset_index(drop=True)
    data_plot['result'] =result[:100]

    plt.figure(figsize=(50,30))
    data_plot.plot()
   
    plt.title('score: %f'%score)
    plt.legend()
    #plt.show()
    
    data = pre_data()
    pre_df =data.copy()
    for col in ['linecodeparentid','linecodename','pol_name','pod_name','vessel_name']:
        gle =LabelEncoder()
        gle.fit(pre_df[col])
        pre_df[col] = gle.transform(pre_df[col])
    pre_df =pre_df[pre_df['etd']> datetime.datetime.now()]
    x = pre_df.drop(['price','etd'],axis=1)
    data['pre_50']=model.predict(x.values)
    od.DataFrame(data).persist ('buy_maersk_50price',overwrite = True,odps=odps)
    

if __name__ == '__main__':
    tree_reg = tree.DecisionTreeRegressor() 
    prediction_to_odps(tree_reg)