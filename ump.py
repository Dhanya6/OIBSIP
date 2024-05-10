
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly .express as px
     
data = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')
data
     

# checking dataset information
data.info()
     

# describing the dataset
data.describe()
     

# check null/missing values
data.isnull().sum()
     

# rename columns
data.columns = ['States','Date','Frequency','Estimated Unemployment Rate',
                'Estimated Employed','Estimated Labour Participation Rate',
                'Region','Longitude','Latitude']
     

# analysing top rows of dataset
data.head(5)
data.tail(5)
    
    # plotting histplot

plt.figure(figsize=(6,6))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate",hue='Region',data=data)
plt.show()
     

# dashboard to analyze the unemployment rate of each indian state 
     
# plotting sunburst
unemployment = data[['States','Region','Estimated Unemployment Rate']]
figure = px.sunburst(unemployment,path=['Region','States'],
                     values='Estimated Unemployment Rate',
                     width=800,height=600, color_continuous_scale='gray',
                     title="UNEMPLOYMENT RATE IN INDIA")
figure.show()
     


     