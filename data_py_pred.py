from datetime import datetime, timedelta
import cx_Oracle
import warnings
warnings.filterwarnings("ignore")

cx_Oracle.init_oracle_client(lib_dir=r"/Users/kaka/Downloads/instantclient_18_1")

import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
import calendar
from datetime import datetime

# 获取当前日期
current_date = datetime.now()
# 获取当前月份的最后一天
last_day = calendar.monthrange(current_date.year, current_date.month)[1]
# 生成最后一天的日期字符串形式
last_day_str = f"{current_date.year}-{current_date.month:02d}-{last_day:02d}"

from datetime import datetime, timedelta


def last_days_of_month_and_six_months_ago(date):
    # 获取当前月份的最后一天
    last_day_current_month = date.replace(day=1, month=date.month + 1)

    # 计算6个月前的日期
    six_months_ago = date - timedelta(days=180)

    # 获取6个月前月份的最后一天
    last_day_six_months_ago = six_months_ago.replace(day=1, month=six_months_ago.month + 1)

    return last_day_current_month.strftime("%Y-%m-%d"), last_day_six_months_ago.strftime("%Y-%m-%d")




class SkuDemodsPred(object):
    def __init__(self):

        self.current_date = datetime.now()
        self.now_time_future, self.end_time_train = \
            last_days_of_month_and_six_months_ago(self.current_date)
        self.lags = 24
        self.steps =180
        self.batch_size = 200

    def mysql_info(self):
        """数据上传"""
        username = 'nppbuf'
        password = 'Svwnppbuf321'
        hostname = '10.122.6.59'
        port = 1521
        service_name = 'b2bt'
        dsn = cx_Oracle.makedsn(hostname, port, service_name=service_name)
        # 连接 Oracle 数据库
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
        return connection

    def data_preprocessing(self,data):
        data.columns = ['零件号', '仓库代码', '年月', '需求流', '需求数量', 'ABC需求频次分类',
                          'ABC价格分类', 'ABC需求数量分类', '零件号第四位','组装包','是否预测']
        data=data[data['是否预测']=='Y']
        data['日期'] = pd.to_datetime(data['年月'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        data = data.groupby(['零件号', '仓库代码', '需求流', '日期'])['需求数量'].sum().reset_index()
        data['日期'] = pd.to_datetime(data['日期'])
        """补零"""
        date_range = pd.date_range(start='2022-01-01', end=self.now_time_future, freq='D')
        # 创建空的DataFrame，准备存储填充后的结果
        filled_df = pd.DataFrame()
        # 针对每个 SKU 进行填充操作
        for sku, group in data.groupby(['零件号', '仓库代码', '需求流']):
            sku_group = group.set_index('日期').reindex(date_range, fill_value=0).reset_index()
            sku_group['零件号'] = sku[0]
            sku_group['仓库代码'] = sku[1]
            sku_group['需求流'] = sku[2]
            filled_df = filled_df.append(sku_group, ignore_index=True)
        return filled_df

    def feature_processing(self,data,end_time):
        """生成列维度时间序列"""
        data = self.data_preprocessing(data)
        data['合并列'] = data['零件号'] + '_' + data['仓库代码'] + '_' + data['需求流']
        # 去除多余的列名
        data = data.drop(columns=['零件号', '仓库代码', '需求流'])
        data = data.set_index(['index', '合并列'])['需求数量'].unstack()
        data.columns.name = None
        data = data.reset_index()
        new_df = data.copy().rename(columns={'index': 'date'})
        new_df['date'] = pd.to_datetime(new_df['date'], format='%Y-%m-%d')
        new_df = new_df.set_index('date')
        new_df = new_df.asfreq('D')
        new_df = new_df.sort_index()
        data_train = new_df[new_df.index < end_time].copy()
        data_test = new_df[new_df.index >= end_time].copy()
        return data_train,data_test

    def demods_groby_month(self,data,type):
        """天级别聚合month"""
        data_prs = data.reset_index()
        data_prs = data_prs.rename(columns={'index': 'date'})
        data_prs["year"] = pd.to_datetime(data_prs['date']).dt.year.astype(int)
        data_prs["month"] = pd.to_datetime(data_prs['date']).dt.month.astype(int)
        data_prs_info = data_prs.groupby(['year', 'month']).sum().reset_index()
        data_prs_info = data_prs_info.set_index(['year', 'month']).stack()
        data_prs_info = data_prs_info.rename_axis(index=['year', 'month', '零件号'])
        data_prs_info.name = type
        data_prs_info = data_prs_info.reset_index()
        data_prs_info[['零件号', '仓库代码', '需求流']] = data_prs_info['零件号'].str.split('_', expand=True).reset_index(drop=True)
        data_prs_info.columns = ['year', 'month', 'hostpartid', 'pred_values', 'hostlocid', 'dshostid']
        data_prs_info['pred_values'] = data_prs_info['pred_values'].astype(float)
        return data_prs_info[[ 'hostpartid', 'hostlocid', 'dshostid','year', 'month','pred_values']]


    def model_pred(self, data,type):
        """批量训练模型"""
        if type =='train':
            end_time = self.end_time_train
        else:
            end_time = self.now_time_future

        data_train, data_test  = self.feature_processing(data,end_time)
        sku_list = data_train.columns
        # 创建空的字典，用于存储每个时间序列的预测模型
        forecasters = {}
        # 循环遍历每批次SKU，分别训练预测模型
        for i in range(0, len(sku_list), self.batch_size):
            batch_skus = sku_list[i:i + self.batch_size]
            for sku in batch_skus:
                forecaster = ForecasterAutoreg(
                    regressor=Ridge(random_state=123),
                    lags=self.lags,
                )
                # 拟合模型
                forecaster.fit(y=data_train[sku])
                forecasters[sku] = forecaster
            # print(f"Finished training batch {i // self.batch_size + 1}/{len(sku_list) // self.batch_size + 1}")
        # 进行未来预测
        predictions = pd.DataFrame()
        for sku, forecaster in forecasters.items():
            forecast = forecaster.predict(steps=self.steps)
            predictions[sku] = forecast
        predictions[predictions < 0.1] = 0
        predictions = self.demods_groby_month(predictions,'pred_values')

        his_info = self.demods_groby_month(data_train, 'pred_values')
        thr_std = his_info.groupby(['hostpartid', 'hostlocid', 'dshostid'])['pred_values'].std().reset_index()
        thr_std['pred_values'] = [x * 3 for x in thr_std['pred_values']]
        full_pred_info = pd.merge(predictions, thr_std, on=['hostpartid', 'hostlocid', 'dshostid'], how='left')
        full_pred_info['pred_values'] = full_pred_info.apply(
            lambda row: row['pred_values_y'] if row['pred_values_x'] > row['pred_values_y'] else row['pred_values_x'],
            axis=1)

        full_pred_info = full_pred_info.drop(['pred_values_x', 'pred_values_y'], axis=1)
        pred_output = full_pred_info.append(his_info)
        pred_output['pred_values'] = pred_output['pred_values'].round(2)
        return pred_output

    def main(self,data,type):
        conn  = self.mysql_info()
        predictions = self.model_pred(data,type)
        # 定义游标
        # 将numpy.int64类型的数据转换为int类型
        predictions['pred_values'] = predictions['pred_values'].astype(float)
        try:
            # 定义游标
            cursor = conn.cursor()
            # 执行插入操作
            sql = "INSERT INTO nppbuf.T_DD_FORECAST_DETAIL_test (hostpartid, hostlocid, dshostid, year, month, pred_values) VALUES (:1, :2, :3, :4, :5, :6)"
            for index, row in predictions.iterrows():
                cursor.execute(sql, (
                row['hostpartid'], row['hostlocid'], row['dshostid'], row['year'], row['month'], row['pred_values']))

            # 提交事务
            conn.commit()

            # 关闭游标和连接
            cursor.close()
            conn.close()
        except:
            predictions.to_csv('111.csv',encoding ='utf-8')
            raise


