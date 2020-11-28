# Databricks notebook source
# Montreal's traffic collisions - Supervised machine learning: classification

# Project topic: To apply a supervised machine learning classifaction algorithm 
# against a public dataset containing Montreal's traffic collision records from 2012 to 2019
# to predict accident's severity (Major/Minor) 
 
# Dataset documentation: https://donnees.montreal.ca/ville-de-montreal/collisions-routieres
# Dataset documentation (2): https://saaq.gouv.qc.ca/donnees-ouvertes/rapports-accident/rapports-accident-documentation.pdf
# Features: Max Speed, number of vehicles, time of the day, day of the week, month, road surface, pedestrian/cyclist
# Objective: try to predict the severity of an accident happening in a given date with enough features to analyze
# Accuracy 79% using Gradient boosting classifiers

import json
import base64
import requests
import numpy as np
import pandas

#Montreal's road collision dataset API - Bring only the data I want

BASE_URL = 'https://data.montreal.ca/api/3/action/datastore_search_sql?sql='

def dbfs_rpc():
  """ Helper function to perfrom API request, request/response is encoded/decoded as JSON """
 
  sql = '''SELECT "CD_ETAT_SURFC", "HEURE_ACCDN", "DT_ACCDN", "NB_VEH_IMPLIQUES_ACCDN", "VITESSE_AUTOR", "GRAVITE", 
           "CD_ECLRM", "CD_GENRE_ACCDN"
           FROM "05deae93-d9fc-4acb-9779-e0942b5e962f" 
           WHERE "VITESSE_AUTOR" > 0 AND "HEURE_ACCDN" <> 'Non précisé' 
           AND "CD_ETAT_SURFC" <> 99 AND "CD_ETAT_SURFC" IS NOT NULL
           AND "CD_ECLRM" IS NOT NULL AND "CD_GENRE_ACCDN" IS NOT NULL           
           '''
  #sql = ''
  response = requests.post(
    BASE_URL + sql
  )
  return response.json()


# COMMAND ----------

#Load and parse Json results from API
x = dbfs_rpc()
y = json.loads(json.dumps(x))

#Creates a new Python dictionary from data at /result/records level
w = y.get('result', {}).get('records', {})

#Show small extract of downloaded data
pandas.DataFrame(w).head()

# COMMAND ----------

#Show unique values of Severity (target varible)
pandas.DataFrame(w).GRAVITE.value_counts()

# COMMAND ----------

#Prepare dataset to be analyzed 

#- Recategorizing Severity(Gravité)
# ---> Dommages matériels* = Minor
# ---> 'Léger','Grave', 'Mortel' = Major

#- Extract numeric month of date
#- Extracting numeric hour
#- Extracting numeric day of the week

road_col = pandas.DataFrame(columns=['month', 'hour', 'week_day','number_veh','max_speed','surface','pedestrian_cyclist', 'major_severity'])

i = 0
for obj in w:
  month = pandas.to_datetime(obj['DT_ACCDN']).month
  hour = int(obj["HEURE_ACCDN"][0:2])
  weekd_day = pandas.to_datetime(obj['DT_ACCDN']).dayofweek #It is assumed the week starts on Monday, which is denoted by 0 and ends on Sunday which is denoted by 6.
  number_veh = int(obj["NB_VEH_IMPLIQUES_ACCDN"])
  max_speed = int(obj["VITESSE_AUTOR"])
      
  if int(obj["CD_GENRE_ACCDN"]) in {32,33}:
    pedestrian_cyclist = 1
  else:
    pedestrian_cyclist = 0      
    
  surface = int(obj["CD_ETAT_SURFC"])
  major_severity_strings = {'Léger','Grave', 'Mortel'}
  if obj["GRAVITE"] in major_severity_strings:
    major_severity = 1
  else:
    major_severity = 0

  road_col.loc[i] = {"month":month, "hour":hour, "week_day":weekd_day, "number_veh":number_veh, "max_speed":max_speed, "surface":surface, "pedestrian_cyclist":pedestrian_cyclist, "major_severity":major_severity}

  i = i + 1
  
road_col.shape

# COMMAND ----------

#Show unique values of Severity (target varible)
road_col.major_severity.value_counts()

# COMMAND ----------

#Data exploration / Visualizations

import seaborn as sns
import matplotlib.pyplot as plt 
plt.rc("font", size=8)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x='major_severity', data=road_col, palette='hls')
plt.show()

# Imbalanced classes on our dataset

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pandas.crosstab(road_col.week_day,road_col.major_severity).plot(kind='bar', stacked=True)
# MAGIC plt.xlabel('Day of the week')
# MAGIC plt.ylabel('Collisions')
# MAGIC 
# MAGIC # Day of the week does not seem to be a good predictor of the outcome variable

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pandas.crosstab(road_col.hour,road_col.major_severity).plot(kind='bar', stacked=True)
# MAGIC plt.xlabel('Hour of the day')
# MAGIC plt.ylabel('Collisions')
# MAGIC 
# MAGIC # Hour of the day seems to be a better predictor of the outcome variable

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pandas.crosstab(road_col.max_speed,road_col.major_severity).plot(kind='bar', stacked=True)
# MAGIC plt.xlabel('Max Speed')
# MAGIC plt.ylabel('Collisions')
# MAGIC 
# MAGIC # Max Speed seems to be a good predictor of the outcome variable

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pandas.crosstab(road_col.number_veh,road_col.major_severity).plot(kind='bar', stacked=True)
# MAGIC plt.xlabel('Number vehicles')
# MAGIC plt.ylabel('Collisions')
# MAGIC 
# MAGIC # Number vehicles seems to be a very good predictor of the outcome variable

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pandas.crosstab(road_col.month,road_col.major_severity).plot(kind='bar', stacked=True)
# MAGIC plt.xlabel('Month')
# MAGIC plt.ylabel('Collisions')
# MAGIC 
# MAGIC # Month of the year seems to be a good predictor of the outcome variable

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC pandas.crosstab(road_col.pedestrian_cyclist,road_col.major_severity).plot(kind='bar', stacked=True)
# MAGIC plt.xlabel('pedestrian_cyclist')
# MAGIC plt.ylabel('Collisions')
# MAGIC 
# MAGIC # The presence of a pedestrian/cyclist seems to be a very good predictor of the outcome variable

# COMMAND ----------

# Splitting data into 7 features and labels (major_severity)
# using the rule 85% Train - 15% Test

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

y = road_col.major_severity.astype('int')
x = road_col.drop('major_severity',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

x_train.head()

# COMMAND ----------

# Train dataset
x_train.shape

# COMMAND ----------

# Train dataset label counts (major_severity)
y_train.value_counts()

# COMMAND ----------

# Test dataset label counts (major_severity)
y_test.value_counts()

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier

#Trying different Learning rates
#lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
#for learning_rate in lr_list:
#    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, random_state=0)
#    gb_clf.fit(x_train, y_train)

#    print("Learning rate: ", learning_rate)
#    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(x_train, y_train)))
#    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(x_test, y_test)))
    
gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=0)
gb_clf.fit(x_train, y_train)
y_pred = gb_clf.predict(x_test)

print('Accuracy of Gradient Boosting classifier (Imbalanced): {:.2f}'.format(gb_clf.score(x_test, y_test)))
print(metrics.classification_report(y_test, y_pred))

#Feature importance
print('Feature Importance')
feature_imp = pandas.Series(gb_clf.feature_importances_,index=road_col.drop('major_severity',axis=1).columns).sort_values(ascending=False)
print(feature_imp)

# COMMAND ----------

#Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sn

data = confusion_matrix(y_test, y_pred, labels=[0,1])

df_cm = pandas.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (7,5))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,fmt='g', annot_kws={"size": 12})

# COMMAND ----------

# Predicting the severity of an accident happening on 02-jan-2021 at 22h (satuday) involving 1 cars + 1 cyclist in a road surface with melting snow

accident = pandas.DataFrame({"month":[1], "hour":[22], "weekd_day":[5], "number_veh":[1], "max_speed":[60], "surface":[15], "pedestrian_cyclist":[1]})
result = gb_clf.predict(accident)
print('Predicted severity: \n',result)

# COMMAND ----------

# Predicting the severity of an accident happening on 06-march-2021 at 17h (satuday) involving 2 cars in a dry road surface 

accident = pandas.DataFrame({"month":[3], "hour":[17], "weekd_day":[5], "number_veh":[2], "max_speed":[50], "surface":[11], "pedestrian_cyclist":[0]})
result = gb_clf.predict(accident)
print('Predicted severity: \n',result)

# COMMAND ----------


