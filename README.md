# Spitogatos_FG

#The asignment wrote on Python3.6 used IDLE editor and Linux software.

This file is created for giving details of my foundings

At first i tried to clean up the data...my foundings and thesolutions that applied or not applied are given below...
  1) Zero Prices : The records deleted
  2) Extreme Prices : On the rest records applied a Zscore filter and records with Zscore>3 are deleted in order to have better visualisation on data.
  3) Duplicate records : After dropping id,ranking_score,ad_type found dublicate records that have been deleted. More than those records are dublicate in dataset but as every agent record different True/False/NaN values or other discrepancies we can not be sure wherever is dublicated or not. 
  4) NaN agent id : Records found without agent_id and deleted as in order to upload a property an id is given to each agent.
  5) Year of construction : 2155 year is changed to 2022. NaNs returned to Zero. When Renovation year found NaN 
  6) Renovation year : to NaNs i inputed the construction year.
  7) Year of construction & Renovation year ZERO/NAN : 233 found , no action taken.
  8) Year of construction < Renovation year : 23 records found but not been corrected due to uncertainty.
  9) Renovation year : NaNs filled with construction year.
  10) Both columns Year of construction and rennovation year transformed to AGE (positive years) from 2022.
  11) Equipped column : contains only NAN and dropped
  12) Living room : values 40,50,60 checked noe by one and replaced with 4,5,6
  13) Rooms : A record with 255 rooms found on a 1000sqm property. Changed to 25. 
  14) Energy class : Nan filled with ZEROs. The rest returned to decimals: A+=7.5,A=7,B+=6.5,B=6,C=5,D=4,E=3,F=2,G=1
  15) Ad Type : Values replaced with weights from 1 to 4. Simple=1,Up=2,Premium=3,Star=4 based on given ranking criteria.
  16) Floor : Values returned to decimal. Ground-floor=0,Basement=-1,Semi-basement=-0.5,Mezzanine=0.5. 
  17) A column of "Yes" occurences number (counter) of each property added in the dataframe
  18) A column of "NAN" counter of each record added in dataframe as well.

First look at our variables...

                         count unique          top   freq         mean          std          min          25%          50%          75%          max
ranking_score            15020    NaN          NaN    NaN       117.73      31.5087           16         95.5        122.4          143        182.8
geography_name           15020      4  south beach   7189          NaN          NaN          NaN          NaN          NaN          NaN          NaN
sq_meters                15020    NaN          NaN    NaN      196.768      1150.25            1           89          137       243.25       140000
price                    15020    NaN          NaN    NaN       535022       461399         8000       235000       395000       690000      2.7e+06
year_of_construction     15020    NaN          NaN    NaN      75.3835      302.416            0           12           26           45         2022
floor                    14330     15            0   4888          NaN          NaN          NaN          NaN          NaN          NaN          NaN
subtype                  15020     10    apartment   9172          NaN          NaN          NaN          NaN          NaN          NaN          NaN
rooms                    15020    NaN          NaN    NaN      2.88395      1.53089            0            2            3            4           25
no_of_bathrooms          15020    NaN          NaN    NaN      1.67397      1.24737            0            1            1            2           21
...


2 boxplots created on aggregated data in order to show the price behaviour grouped by property type and geography area and a sceond grouped on reserved mode first by geography area and after by type. 

  1st Conclusion : Higher prices occured in Northern Sub and South Beach than in Gentrification area and Beesy neighborhood. The prices also are much more  spreaded ini the first 2 areas with many extreme high values.Also great differences on prices of properties based on type.

  2nd Conclusion : A violin plot created on aggregated data of Ranking_score, Ad_type and Geography_area. No ranking hierarchy found based on Ad_type and correlated with ranking_score. The results based on the plot and on pearson correlation. Both found a weak correlation on the Gentrification Area.

  3rd conclusion : A Regression tried to applied on data for predicting the price on each property.

  1)At first all the fields with categorical data replaced True/False/NaN to 1/0/0.
  2)Geography area hotencoded to each one of the 4 areas
  3)As well as Subtype hotencoded to each one of the given type of properties. 
  4)Above boolean fields added in the dataframe and the 2 columns of subtype and geography area droped.
  5)The dataframe splitted to the Price list and the Variables matrix
  6)The 2 matrices splitted to train sets and predicted set.
  7)Non linear regresion (RandomForestRegression) applied to data to train the set and apply the prediction.
  8)The prediction returned over 75% of accuracy for 80% of minimum price.
  9)Most weighted Variables found...
      Variable: sq_meters            Importance: 0.63
      Variable: year_of_construction Importance: 0.07
      Variable: south beach          Importance: 0.04
      Variable: floor                Importance: 0.03
      Variable: ranking_score        Importance: 0.02
      Variable: renovation_year      Importance: 0.02
      Variable: beesy neighborhood   Importance: 0.02
      Variable: rooms                Importance: 0.01
      Variable: no_of_bathrooms      Importance: 0.01
      Variable: no_of_wc             Importance: 0.01
      Variable: pool                 Importance: 0.01
      Variable: NaNs                 Importance: 0.01
      Variable: Yes                  Importance: 0.01
      Variable: detached             Importance: 0.01
      Variable: maisonette           Importance: 0.01
  10)For a next step and for better predictions we can keep this set of variables (maybe some more that we think that they are important) and run again the algorithm for a better prediction ~79%.





