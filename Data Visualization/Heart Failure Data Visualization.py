# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:29:47 2020

@author: Sahil Bagwe
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:48:45 2020

@author: Sahil Bagwe
"""

import numpy as np
import pandas as pd
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.renderers.default='browser'
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
import os

if not os.path.exists():
    os.mkdir()


chart_studio.tools.set_credentials_file(username='', api_key='')

df = pd.read_csv('')

df['sex'].replace(1,'Male',inplace = True)
df['sex'].replace(0,'Female',inplace = True)
df['DEATH_EVENT'].replace(1,'Heart Attack',inplace = True)
df['DEATH_EVENT'].replace(0,'Alive',inplace = True)
df['anaemia'].replace(1,'Anemic',inplace = True)
df['anaemia'].replace(0,'Non-Anemic',inplace = True)
df['diabetes'].replace(1,'Diabetic',inplace = True)
df['diabetes'].replace(0,'Non-Diabetic',inplace = True)
df['smoking'].replace(1,'Smoker',inplace = True)
df['smoking'].replace(0,'Non-Smoker',inplace = True)
df['high_blood_pressure'].replace(1,'Hypertension',inplace = True)
df['high_blood_pressure'].replace(0,'Other',inplace = True)

df.rename(columns={'DEATH_EVENT':'Patient Status'}, inplace=True)
df.rename(columns={'high_blood_pressure':'High Blood Pressure'}, inplace=True)
df.rename(columns={'creatinine_phosphokinase':'Creatinine Phosphokinase'}, inplace=True)
df.rename(columns={'ejection_fraction':'Ejection Fraction'}, inplace=True)
df.rename(columns={'platelets':'Platelets'}, inplace=True)
df.rename(columns={'serum_sodium':'Serum Sodium'}, inplace=True)
df.rename(columns={'serum_creatinine':'Serum Creatinine'}, inplace=True)
df.rename(columns={'age':'Age'}, inplace=True)
df.rename(columns={'anaemia':'Anaemia'}, inplace=True)
df.rename(columns={'diabetes':'Diabetes'}, inplace=True)
df.rename(columns={'sex':'Sex'}, inplace=True)
df.info()

#%% Platelet 
df.loc[df['Platelets'] <150000, 'Platelet Level'] = 'Low'
df.loc[df['Platelets'] >350000, 'Platelet Level'] = 'High'
df.loc[(df['Platelets'] < 350000) & (df['Platelets'] > 150000), 'Platelet Level'] = 'Normal'

ds = df['Platelet Level']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['Platelet Level','Patient Status', 'Count']

fig = px.bar(dx,x="Platelet Level",y='Count',color="Patient Status",barmode="group")
fig.show()



#%% Sex
x=df.groupby("Sex").count()
fig = px.pie(df,names='Sex')
fig.show()

#%% Serum Ceratinine

df.loc[df['Serum Creatinine'] <0.84, 'Creatinine Level'] = 'Low'
df.loc[df['Serum Creatinine'] >1.21, 'Creatinine Level'] = 'High'
df.loc[(df['Serum Creatinine'] < 1.21) & (df['Platelets'] > 0.84), 'Creatinine Level'] = 'Normal'

ds = df['Creatinine Level']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['Creatinine Level','Patient Status', 'Count']

fig = px.bar(dx,x="Creatinine Level",y='Count',color="Patient Status",barmode="group")
fig.show()




#%% Serum Sodium

df.loc[df['Serum Sodium'] <135, 'Sodium Level'] = 'Low'
df.loc[df['Serum Sodium'] >145, 'Sodium Level'] = 'High'
df.loc[(df['Serum Sodium']<=145) & (df['Platelets'] >= 135), 'Sodium Level'] = 'Normal'

ds = df['Sodium Level']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['Sodium Level','Patient Status', 'Count']

fig = px.bar(dx,x="Sodium Level",y='Count',color="Patient Status",barmode="group")
fig.show()





#%% Ejection Fraction Bar Chart

df.loc[df['Ejection Fraction'] <55, 'Ejection Fraction Level'] = 'Low'
df.loc[df['Ejection Fraction'] >65, 'Ejection Fraction Level'] = 'High'
df.loc[(df['Ejection Fraction'] <=65) & (df['Ejection Fraction'] >= 55), 'Ejection Fraction Level'] = 'Normal'

ds = df['Ejection Fraction Level']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['Ejection Fraction Level','Patient Status', 'Count']

fig = px.bar(dx,x="Ejection Fraction Level",y='Count',color="Patient Status",barmode="group")
fig.show()


#%% Ejection Fraction Violin

fig = px.box(df,x='High Blood Pressure',y="Ejection Fraction")
fig.show()

#%% Blood Pressure Pie Chart

ds = df['High Blood Pressure']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['High Blood Pressure','Patient Status', 'Count']
fig = px.pie(dx,values='Count', names='High Blood Pressure')
fig.show()
fig.show()



#%% HyperTension Violin

fig = px.violin(df,x='High Blood Pressure',y='age',color="Patient Status", box=True, points="all",hover_data=df.columns)
fig.show()


#%% Death event

ds = df['Patient Status'].value_counts().reset_index()
ds.columns = ['Patient Status', 'count']
fig1 = px.pie(ds, values='count',  names='Patient Status', title='Patient Status')
fig1.update_layout(title_x = 0.5)
fig1.show()

#%% Anemia

ds = df['anaemia']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['Anaemia','Patient Status', 'Count']
fig = px.pie(dx,values='Count',names='Anaemia')
fig.update_layout(title_x = 0.5)
fig.show()



fig1=px.bar(dx,x="Anaemia",y="Count",color="Patient Status")
fig1.show()

#%% Smoking Pie Chart

ds = df['smoking']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['Smoking','Patient Status', 'Count']
fig = px.pie(dx,values='Count', names='Smoking')
fig.show()
fig.write_html("C:/Users/Sahil Bagwe/Desktop/Python/graphs/smoking_pie.html")
fig.update_layout(title_x = 0.5)
fig.show()

py.plot(fig, filename = 'Smoking Pie',auto_open=False)

#Smoking Violin Chart
fig = px.violin(df,x='smoking',y='age',color="Patient Status", box=True, points="all",hover_data=df.columns)
fig.show()


#%% Diabetes Pie Chart

ds = df['diabetes']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['diabetes','Patient Status', 'count']

fig = px.pie(dx,values='count', names='diabetes')
fig.show()

fig.write_html("C:/Users/Sahil Bagwe/Desktop/Python/graphs/diabetes_pie.html")
py.plot(fig, filename = 'disabetes_pie',auto_open=False)

fig = px.violin(df,x='diabetes',y='age',color="Patient Status", box=True, points="all",hover_data=df.columns)
fig.show()

#%% Ceratinine Phosphokinase

df.loc[df['Creatinine Phosphokinase'] <22, 'CPK Level'] = 'Low'
df.loc[df['Creatinine Phosphokinase'] >198, 'CPK Level'] = 'High'
df.loc[(df['Creatinine Phosphokinase'] < 120) & (df['Platelets'] > 10), 'CPK Level'] = 'Normal'

ds = df['CPK Level']
ds = ds.to_frame()
ds['Patient Status'] = df['Patient Status']
dx = ds.value_counts().reset_index()
dx.columns = ['CPK Level','Patient Status', 'Count']

fig = px.bar(dx,x="CPK Level",y='Count',color="Patient Status",barmode="group")
fig.show()


#%% Histogram Age

fig = px.histogram(df,x='age',nbins=50,color='Patient Status',barmode = 'relative',title=('Age & Heart Attack Distribution'))
fig.update_layout(title_x = 0.5)
fig.show()




#%% Anaemia Pie Plot


ds = df['anaemia'].value_counts().reset_index()
ds.columns = ['anaemia', 'count']
fig1 = px.pie(ds, values='count',  names='anaemia', title='Anaemia bar chart')
fig1.update_layout(title_x = 0.5)
fig1.show()



#%% Anemia Violin    
fig1 = px.violin(df,x='Anaemia',y='Age',color="Patient Status", box=True, points="all",hover_data=df.columns)
fig1.show()    

%% Heat map

sns.displot(df, x ='age',hue='diabetes',multiple="stack" )
fig =  px.histogram(df,x='age',color ='sex', color_discrete_map = {0:'green',1:'blue'})
fig.show()

for col in df.columns:
    print(col, str(round(100* df[col].isnull().sum() / len(df), 2)) + '%')
# fig = plt.figure(figsize = (20,20))
# sns.heatmap(df.corr(),cmap="YlGnBu", annot = True)

