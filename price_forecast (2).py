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


# 寻找区间
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def pre_data():
    
    data =gdfo.select_table_to_dl()
    #data =pd.read_csv('msk_price_online.csv',encoding ='utf-8')
    data = data[['from_terminal_id', 'discharging_terminal_id', 'etd', 'price_unit', 'price', 'valid_from', 'linecodename','linecodeparentid']]
    data['etd'] = pd.to_datetime(data['etd']).dt.floor('d')
    data['valid_from'] = pd.to_datetime(data['valid_from']).dt.floor('d')
    data = data.drop_duplicates()
    data['price']=data['price'].astype('float')
    dataset_price = pd.DataFrame(data.groupby(['from_terminal_id', 'discharging_terminal_id', 'price_unit', 'etd', 'valid_from'])['price'].agg('max','min'))
    dataset_price_day = dataset_price.reset_index()
    maersk_price_dataset = pd.merge(dataset_price_day, data,on=['from_terminal_id', 'discharging_terminal_id', 'price_unit', 'etd', 'valid_from'])
    
    
    data_columns = maersk_price_dataset.drop(['linecodename', 'linecodeparentid', 'price_x'], axis=1)
    dataset_price_columns = pd.DataFrame(data_columns.groupby(['from_terminal_id', 'discharging_terminal_id', 'price_unit', 'etd'])['price_y'].agg(percentile(50)))
    dataset_price_col_index = dataset_price_columns.reset_index()
    
    dataset_price_col_index['price_type'] = dataset_price_col_index['price_unit'].apply(lambda x: x[:2])
    dataset_price_col_index['etd'] = pd.to_datetime(dataset_price_col_index['etd']).dt.floor('d')
    # dataset_price['etd_days']= (dataset_price['etd'] -pd.to_datetime('2020-01-01')).map(lambda x:x.days)
    dataset_price_col_index['etd_days'] = dataset_price_col_index['etd'].dt.day
    dataset_price_col_index['etd_month'] = dataset_price_col_index['etd'].dt.month
    dataset_price_col_index['etd_q'] = (dataset_price_col_index['etd'].dt.to_period('Q')).apply(lambda x: str(x)[-2:]).str.replace('Q', '')
    
    return dataset_price_col_index
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)

def prediction_to_odps(clf):
    odps = gdfo.connect_odps()
    dataset =pre_data()

    x =dataset.drop(['price_y','etd','price_unit'],axis =1)
    y =dataset['price_y']
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state =1)
    
    
    model =clf.fit(x_train,y_train)

    
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    mape = MAPE(y_test, result)
    print('score: %f,mape: %f'%(clf.score(x_test, y_test),MAPE(y_test, result)))
    
    data_plot =pd.DataFrame()

    data_plot['y_test'] =y_test[:100]
    data_plot = data_plot.reset_index(drop=True)
    data_plot['result'] =result[:100]

    plt.figure(figsize=(50,30))
    data_plot.plot()
   
    plt.title('score: %f'%score)
    plt.legend()
    
    fu_data =dataset[dataset['etd']> datetime.datetime.now().strftime("%Y-%m-%d")]
    
    fu_data = fu_data.drop_duplicates()
    fu_data['price_type'] = fu_data['price_unit'].apply(lambda x: x[:2])
    fu_data['etd'] = pd.to_datetime(fu_data['etd']).dt.floor('d')
    fu_data['etd_days'] = fu_data['etd'].dt.day
    fu_data['etd_month'] = fu_data['etd'].dt.month
    fu_data['etd_q'] = (fu_data['etd'].dt.to_period('Q')).apply(lambda x: str(x)[-2:]).str.replace('Q', '')
    x = fu_data[['from_terminal_id', 'discharging_terminal_id', 'price_type', 'etd_days', 'etd_month', 'etd_q']]
    
    fu_data['pre_price'] = model.predict(x.values)
    fu_data['created_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_to_odps = fu_data[['from_terminal_id', 'discharging_terminal_id', 'price_unit', 'etd', 'etd_days', 'etd_month', 'etd_q', 'pre_price','created_time']]
    data_to_odps = data_to_odps.drop_duplicates()
    data_to_odps.to_csv('12.csv',encoding ='utf-8')
    #od.DataFrame(data_to_odps).persist ('ps_threshold_prediction_daily',overwrite = False,odps=odps)
    
if __name__ == '__main__':
    tree_reg = tree.DecisionTreeRegressor()
    prediction_to_odps(tree_reg)



    航线属性
        一级航线：根据各航线指数（scif）确定市场火热状态，淡旺季（0，1）

        二级航线：三级航线数量 （是否主力航线）

        三级航线   

    港口属性
        起始港  
            目的港 
        所属国家  
        航线距离：根据http://port.sol.com.cn/licheng.asp 匹配大港之间的距离，小港用经纬度测算离最近的大港距离累加
    
    时间属性  
        #etd 周数     
　　　     etd 月份  
        # 节假日  
　　　     距离出发日期的天数 gap

    价格属性  

    相同pol-pod 历史价格特征：
　　　     历史提价次数，降价次数，
　　　     历史价格序列的平均值，标准差，最大值，最小值
　　　     首版价格
　　　     首版价格相对于历史首价格的波动幅度（比例）
　　　     价格波动平均间隔时间
　　　     价格波动平均次数  
