# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:58:22 2019

@author: catherine
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

##读数据集
data = pd.read_csv('cbg_patterns.csv')
##标称属性 共7个
df1 = data.ix[:,['visitor_home_cbgs','visitor_work_cbgs','related_same_day_brand','related_same_month_brand','top_brands','popularity_by_hour',
          'popularity_by_day']]
##数值属性 共6个
df2 = data.ix[:,['census_block_group','date_range_start','date_range_end','raw_visit_count','raw_visitor_count','distance_from_home']]

#对标称属性，给出每个可能取值的频数
#以visitor_home_cbgs为例
print(df1['visitor_home_cbgs'].value_counts())

##对数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。
print(df2.max(),df2.min(),df2.mean(),df2.median(),df2.quantile(0.25),len(df2)- df2.count())

##以census_block_group为例,绘制直方图
##剔除缺失部分
ddf2 = df2['census_block_group'].dropna()  
plt.hist(ddf2,bins=100)
plt.show()
plt.savefig('census_block_group_hist_1.png')

##QQ图
sorted_ = np.sort(ddf2)
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(ddf2, dist="norm", plot=plt)
plt.show()
plt.savefig('census_block_group_QQ_1.png')


##盒图
plt.boxplot(ddf2)
plt.ylabel('census_block_group')
plt.legend()
plt.show()
plt.savefig('census_block_group_box_1.png')


#以最高频率值来填补缺失值
#插值interpolate
#使用most_frequent
#1 直方图
plt.hist(df2['census_block_group'].fillna(df2['census_block_group'].interpolate(missing_values = 'NaN', strategy = 'most_frequent', axis = 0, verbose = 0, copy = True)),bins=100)
plt.show()
plt.savefig('census_block_group_hist_2.png')
#2 QQ图

sorted_ = np.sort(df2['census_block_group'].fillna(df2['census_block_group'].interpolate(missing_values = 'NaN', strategy = 'most_frequent', axis = 0, verbose = 0, copy = True)))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(df2['census_block_group'], dist="norm", plot=plt)
plt.show()
plt.savefig('census_block_group_QQ_2.png')

#3 盒图
plt.boxplot(df2['census_block_group'].fillna(df2['census_block_group'].interpolate(missing_values = 'NaN', strategy = 'most_frequent', axis = 0, verbose = 0, copy = True)))
plt.ylabel('census_block_group')
plt.legend()
plt.show()
plt.savefig('census_block_group_box_2.png')

#利用属性间的关系
#使用热点图计算和census_block_group相关的属性raw_visit_count
sns.heatmap(df2.corr(),annot=True)

known_price = df2[df2['raw_visitor_count'].notnull()]
unknown_price = df2[df2['raw_visitor_count'].isnull()]
x = known_price[['date_range_start']]
y = known_price[['raw_visitor_count']]
t_x = unknown_price[['date_range_start']]
fc=RandomForestClassifier()
fc.fit(x,y)
pr=fc.predict(t_x)
df2.loc[df2.raw_visitor_count.isnull(),'raw_visitor_count'] = pr
#直方图
plt.hist(df2['raw_visitor_count'],bins=100)
plt.show()
plt.savefig('raw_visitor_count_hist_3.png')
#QQ图
sorted_ = np.sort(df2['raw_visitor_count'])
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(df2['date_range_start'])
plt.show()
plt.savefig('raw_visitor_count_QQ_3.png')
#盒图
plt.boxplot(df2['date_range_start'])
plt.ylabel('date_range_start')
plt.legend()
plt.show()
plt.savefig('raw_visitor_count_box_3.png')

#通过数据对象之间的相似性来填补缺失值
#插值-mean
#1 直方图
plt.hist(df2['census_block_group'].fillna(df2['census_block_group'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)),bins=100)
plt.show()
plt.savefig('census_block_group_hist_4.png')
#2 QQ图

sorted_ = np.sort(df2['census_block_group'].fillna(df2['census_block_group'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(df2['census_block_group'], dist="norm", plot=plt)
plt.show()
plt.savefig('census_block_group_QQ_4.png')

#3 盒图
plt.boxplot(df2['census_block_group'].fillna(df2['census_block_group'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)))
plt.ylabel('census_block_group')
plt.legend()
plt.show()
plt.savefig('census_block_group_box_4.png')


