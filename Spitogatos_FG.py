# Spitogatos Asignment #
#
# Filippos Georgoulis
#
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats  import pearsonr
from scipy.stats import zscore
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

#set working dirrectory
os.chdir('/home/filippos/Desktop/Spitogatos')
#load csv as dataframe
data=pd.read_csv('assignment_rev2.csv', header=0)
#set all columns visible in results
pd.set_option('max_columns',None)
#first 5 rows
data.head()
#check for duplicate id's
data['id'].is_unique
#set width for printing results
pd.set_option('display.width', 1000)

#DATA CLEARANCE
Cdata=data  #15447 rows

#Floors to decimals
Cdata.floor.replace({'ground-floor':0,'basement':-1,'semi-basement':-0.5,'mezzanine':0.5},inplace=True)

#112//id=42768159//score=111.5//sqm=935///rooms=255 (?)...
Cdata['rooms'].replace(255,25,inplace=True) #replace 255 rooms to 25

#living_rooms values 40,50,60 clear up (40,50,60:4,5,6) after check
Cdata['living_rooms'].replace({40:4,50:5,60:6},inplace=True)

#add a NaN count column
Cdata['NaNs']=Cdata.isnull().sum(axis=1)
#add a True count column
Cdata['Yes']=Cdata[Cdata==True].count(axis=1)

#DROP DUBLICATES

#drop id,ranking_score,ad_type that makes all rows unique
DUPdata=Cdata.drop(columns=['id','ranking_score','ad_type'],inplace=False)
#drop 57 duplicates from Cdata 
Cdata=Cdata[~DUPdata.duplicated()]
#delete dataframe
del DUPdata

#CLEAN EXTREME PRICES

#drop ZERO values and UNDER 1000E
Cdata=Cdata[Cdata.price>1000]
#15365 rows

#drop extreme values
Cdata=Cdata[np.abs(zscore(Cdata.price))<3] #which drops high only prices
#15124 rows

#drop na agent_id (104 records)
Cdata=Cdata[~Cdata.agent_id.isna()]
#15020 rows

#drop equipped column filled of NaNs
Cdata.drop(columns='equipped',inplace=True)

#replace construction year 2155 to 2022
Cdata.year_of_construction.replace({2155:2022},inplace=True)

#renovation year fillna toconstruction year
Cdata['renovation_year']=Cdata.renovation_year.fillna(Cdata.year_of_construction)

#replace year of construction and renovation year with AGE(from today)
Cdata['year_of_construction']=2022-Cdata.year_of_construction
Cdata['renovation_year']=2022-Cdata.renovation_year.astype(int)

##### FIRST 1 #####

#metrics of dataframe
Cdata.describe(include='all').transpose()

#keep only subtype, geography_name,price
Fdata=Cdata[['subtype','geography_name','price']]

# Price per subtype/area
#mean , median, std  Aggregate
Fdata.groupby(['subtype','geography_name']).agg({'price':['mean','median','std','count']}).round()

#Create boxplot Violin per subtype/area
sns.set(style='whitegrid')
sns.boxplot(x='geography_name', y='price', hue='subtype', data=Fdata)
plt.show()

# Price per subtype/area
#mean , median, std  Aggregate
Fdata.groupby(['geography_name','subtype']).agg({'price':['mean','median','std','count']}).round()

#Create boxplot Violin per subtype/area
sns.set(style='whitegrid')
sns.boxplot(x='subtype', y='price', hue='geography_name', data=Fdata)
plt.show()


#####  SECOND 2   #####

Sdata=Cdata

#list of NaN counted for each column
nancount=Sdata.isnull().sum(axis=0)

#names of columns without NaNs
nanlist=nancount[nancount==0]._index

#select all data with no NaNs 
Sdata=Sdata[nanlist]

#plot per geography_name/ad_type of ranking_score
sns.set(style='whitegrid')
sns.violinplot(x='geography_name', y='ranking_score', hue='ad_type', data=Sdata, hue_order=['simple','up','premium','star'])
plt.show()

#correlation between ranking_score and ad_type
RANKdata=Sdata
RANKdata.ad_type=RANKdata['ad_type'].replace({'simple':1,'up':2,'premium':3,'star':4})
x_rank=RANKdata.ranking_score
y_ad=RANKdata.ad_type

#list of geography names
geonames=Sdata.geography_name.unique()
for i in geonames:
    sample_x=RANKdata[RANKdata.geography_name==i].ranking_score
    sample_y=RANKdata[RANKdata.geography_name==i].ad_type
    sample_p=pearsonr(sample_x,sample_y)
    print('Pearson correlation between Ranking_score and Ad_type in %s is %f .'%(i,sample_p[0]))

'''
Pearson correlation between Ranking_score and Ad_type in northern sub is 0.091698 .
Pearson correlation between Ranking_score and Ad_type in south beach is 0.028341 .
Pearson correlation between Ranking_score and Ad_type in gentrification area is 0.139208 .
Pearson correlation between Ranking_score and Ad_type in beesy neighborhood is 0.032055 .
'''

#####  THIRD 3   #####
#
#Regression

Tdata=Cdata

#ad_type to int
Tdata['ad_type']=Tdata.ad_type.replace({'simple':1,'up':2,'premium':3,'star':4})
#energy_class to decimal
Tdata.energy_class=Tdata.energy_class.replace({'aplus':7.5,'bplus':6.5,'a':7,'b':6,'c':5,'d':4,'e':3,'f':2,'g':1})

#all NaNs to zero
Tdata=Tdata.fillna(0)
#True/False to boolean 1/0
Tdata=Tdata*1

#Hot encode categorical data of geography_name
Tdata1=pd.get_dummies(Cdata.geography_name)
#Hot encode categorical data of subtype
Tdata2=pd.get_dummies(Cdata.subtype)

#keep non boolean data
TTnames=Tdata.columns
#Join 3 dataframes Cdata&GeogEnc&SubEnc
TT=Tdata[TTnames].join(Tdata1).join(Tdata2) 
#Drop id's
TT.drop(columns=['id','agent_id','geography_name','subtype'],inplace=True)

#Tprice for PREDICTION
Tprice=np.array(TT['price'])
#Drop Price from TT for variables
TTvars=TT.drop(columns='price')
#Name list of variables
Tnames=list(TTvars.columns)
#Convert to array
Tvars=np.array(TTvars)

#Split data into training and testing set

#import libraries
from sklearn.model_selection import train_test_split
#split data
train_T,test_T,train_P,test_P=train_test_split(Tvars,Tprice,test_size=0.25,random_state=42)


print('Training Features Shape:',train_T.shape)
print('Training Price Shape:',train_P.shape)
print('Testing Features Shape:',test_T.shape)
print('Testing Price Shape:',test_P.shape)


#Baseline predictiions -historical averages
baseline_preds=test_P*0.8 #baseline_preds=test_T[:,Tnames.index('average')]

#Baseline errors, display average baseline error
baseline_errors=abs(baseline_preds - test_P)

#Set the goal prediction
print('Average baseline error:', round(np.mean(baseline_errors),2))

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=100,random_state=42)
rfr.fit(train_T,train_P)

#use the forest's predict method on the test data
predictions=rfr.predict(test_T)
#calculate the absolute errors
errors=abs(predictions-test_P)
#print mean absolute error (MAE)
print('Mean Absolute Error:',round(np.mean(errors),2),'...')

#calculate mean absolute percentage error (MAPE)
mape=100*(errors/test_P)
#calculate & display accuracy
accuracy=100-np.mean(mape)
print('Accuracy:',round(accuracy,2),'%.')

#numerical feature importances
importances=list(rfr.feature_importances_)
#list of tuples with variable and importance
feature_importances=[(feature,round(importance,2)) for feature,importance in zip(Tnames,importances)]
#sort the feature importances by most important first
feature_importances=sorted(feature_importances,key=lambda x:x[1],reverse=True)
#print feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


#Keep only the important variables and run again the regression
xnames=['sq_meters','year_of_construction','south beach','floor','ranking_score','renovation_year','beesy neighborhood','rooms','no_of_bathrooms','no_of_wc','pool','NaNs','Yes','detached','maisonette']
Tvars=np.array(TTvars[xnames])

#repeat the same steps as above.



#a linear regression applied also but with very bad results as it was more than expected
'''
#price column of dataframe
Tprice
#Value Names
Tnames
#Values
Tvars

x=TTvars.astype(float)

#create function to calculate stats
def get_stats():
    results=sm.OLS(Tprice,x).fit()
    print(results.summary())
#apply our stats
get_stats()
'''
