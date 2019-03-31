# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:02:05 2019

@author: catherine
"""

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
data = pd.read_csv('winemag-data_first150k.csv')
##标称属性 共8个
df1 = data.ix[:,['country','description','designation','province','region_1','region_2','variety','winery']]
##数值属性 共2个
df2 = data.ix[:,['points','price']]

#对标称属性，给出每个可能取值的频数
#以province为例
print(df1['province'].value_counts())

##对数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。
print(df2.max(),df2.min(),df2.mean(),df2.median(),df2.quantile(0.25),len(df2)- df2.count())

##以census_block_group为例,绘制直方图
##剔除缺失部分
ddf2 = df2['price'].dropna()  
plt.hist(ddf2,bins=100)
plt.show()
plt.savefig('price_hist_1.png')

##QQ图
sorted_ = np.sort(ddf2)
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(ddf2, dist="norm", plot=plt)
plt.show()
plt.savefig('price_QQ_1.png')


##盒图
plt.boxplot(ddf2)
plt.ylabel('price')
plt.legend()
plt.show()
plt.savefig('price_box_1.png')


#以最高频率值来填补缺失值
#插值interpolate
#使用most_frequent
#1 直方图
plt.hist(df2['price'].fillna(df2['price'].interpolate(missing_values = 'NaN', strategy = 'most_frequent', axis = 0, verbose = 0, copy = True)),bins=100)
plt.show()
plt.savefig('price_hist_2.png')
#2 QQ图

sorted_ = np.sort(df2['price'].fillna(df2['price'].interpolate(missing_values = 'NaN', strategy = 'most_frequent', axis = 0, verbose = 0, copy = True)))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(df2['price'], dist="norm", plot=plt)
plt.show()
plt.savefig('price_QQ_2.png')

#3 盒图
plt.boxplot(df2['price'].fillna(df2['price'].interpolate(missing_values = 'NaN', strategy = 'most_frequent', axis = 0, verbose = 0, copy = True)))
plt.ylabel('price')
plt.legend()
plt.show()
plt.savefig('price_box_2.png')

#利用属性间的关系
#数值属性只有points和price，故只能用points预测price
known_price = df2[df2['price'].notnull()]
unknown_price = df2[df2['price'].isnull()]
x = known_price[['points']]
y = known_price[['price']]
t_x = unknown_price[['points']]
fc=RandomForestClassifier()
fc.fit(x,y)
pr=fc.predict(t_x)
df2.loc[df2.price.isnull(),'price'] = pr
#直方图
plt.hist(df2['price'],bins=100)
plt.show()
plt.savefig('price_hist_3.png')
#QQ图
sorted_ = np.sort(df2['price'])
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(df2['points'])
plt.show()
plt.savefig('price_QQ_3.png')
#盒图
plt.boxplot(df2['points'])
plt.ylabel('points')
plt.legend()
plt.show()
plt.savefig('price_box_3.png')

#通过数据对象之间的相似性来填补缺失值
#插值-mean
#1 直方图
plt.hist(df2['price'].fillna(df2['price'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)),bins=100)
plt.show()
plt.savefig('price_hist_4.png')
#2 QQ图

sorted_ = np.sort(df2['price'].fillna(df2['price'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)))
yvals = np.arange(len(sorted_))/float(len(sorted_))
x_label = stats.norm.ppf(yvals)
plt.scatter(x_label, sorted_)
stats.probplot(df2['price'], dist="norm", plot=plt)
plt.show()
plt.savefig('price_QQ_4.png')

#3 盒图
plt.boxplot(df2['price'].fillna(df2['price'].interpolate(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)))
plt.ylabel('price')
plt.legend()
plt.show()
plt.savefig('price_box_4.png')


