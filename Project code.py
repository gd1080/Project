#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read and load data and create 7 plots 

This script reads .csv files containing environmental data and creates 7 plots
that cna be used to look at seasonality of different variables

Parameters
----------
data: dataframe containing radiation and precip data

infile_name: name of the file containing raw data 

fig_title: The location of the tower station that recorded the data

@author = Gavin Drake
@date = 2023- ***
@license = MIT -- https://opensource.org/licenses/MIT
"""


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


#%% Specify Parameters

infile_name = 'Daily Project Data.csv'

fig_title = 'North Orono, ME'


#%% Load Data

#loading data and removing -9999 values
def readscan(filename):
   data = pd.read_csv(infile_name, comment = '#', parse_dates = ['TIMESTAMP'],
                      index_col = ['TIMESTAMP'], na_values = ('-9999'))
   data = data.resample('D').mean()
   data = data.rename(columns={'P_F': 'Precip', 'TA_F': 'Air_Temp', 'WS_F': 'WS', 
                               'CO2_F_MDS': 'CO2'})
   data.replace([np.nan], 0, inplace = True)
   return data

data = readscan(infile_name)


#%% For Loop 

# Making for loop to add columns to 'data'
columns= ['H', 'LE', 'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT', 'TS', 
          'Precip', 'NETRAD', 'Air_Temp', 'WS', 'CO2']
for c in columns:
     if c in data.columns.tolist():
         pass
     else:
         data[c]=data.filter(like=c+'_', axis=1).mean(axis=1)
data = data[columns]


#%% Calculations 

# Rolling Command
data['7_day_AT'] = data['Air_Temp'].rolling(7, center = False).mean()

data['7_day_Rad'] = data['NETRAD'].rolling(7, center = False).mean()

# creating a start and end date for the water year
startdate=data.loc[(data.index.month==10)&(data.index.day==1)].index[0]

enddate=data.loc[(data.index.month==9)&(data.index.day==30)].index[-1]

data=data[startdate:enddate]


#%% Plot 1 

# Plot scatter plot
fig, ax = plt.subplots()
ax.scatter(data['7_day_AT'],data['7_day_Rad'])
x=data['7_day_AT']
y=data['7_day_Rad']
coefficients = np.polyfit(x, y, 2)

# Plotting the best-fit line
x_fit = np.linspace(min(x), max(x), num=100)
y_fit = coefficients[0]*(x_fit)**2+ coefficients[1] * x_fit + coefficients[2]
plt.plot(x_fit, y_fit, color='red')

# Add x-axis label
ax.set_xlabel('Air Tamp (C)')

# Add y-axis label
ax.set_ylabel('Net Radiation (W mâˆ’2')

# Add plot title
ax.set_title(fig_title)


#%% plot 2

# Plot scatter plot
fig, ax = plt.subplots()
ax.scatter(data['Air_Temp'],data['CO2'])
x=data['Air_Temp']
y=data['CO2']
coefficients = np.polyfit(x, y, 2)

# Plotting the best-fit line
x_fit = np.linspace(min(x), max(x), num=100)
y_fit = coefficients[0]*(x_fit)**2+ coefficients[1] * x_fit + coefficients[2]
plt.plot(x_fit, y_fit, color='red')

# Add x-axis label
ax.set_xlabel('Air Temp (C)')

# Add y-axis label
ax.set_ylabel('Mole Fraction CO2')

# Add plot title
ax.set_title(fig_title)


#%% Monthly CO2 calc

# Add columns (month, water_year,day_of_year)
data['month']=data.index.month
data['water_year']=data.index.year
data.loc[data['month']>9,'water_year']= data.loc[data['month']>9,'water_year'] +1
data['day_of_year']=data.index.dayofyear

# Annual data series
data_annual = data.groupby('water_year')['CO2'].sum()

# Calculations
c = np.mean(data_annual) #avg annual total CO2

# Average monthly calc
c_month = c/12

# New dataframe for monthly data
data_monthly = data.groupby('month')[['day_of_year']].median()

# Creating new CO2 column in new dataframe
data_monthly['CO2'] = np.NAN

for i in range(1,13):
    cmonth = data.loc[data['month']==i,['CO2', 'water_year']]
    monthmean = cmonth.groupby('water_year').sum().mean()
    data_monthly.loc[i,'CO2'] = monthmean[0]
    
    
#%% CO2 Bar Chart

fig, ax = plt.subplots()

# Plot monthly precipitation
ax.bar(data_monthly.index, data_monthly['CO2'], label='Monthly CO2')

# Plot average monthly precipitation
ax.axhline(c_month, ls='-', color='k', label='Average Monthly CO2')

# Add x-axis label
ax.set_xlabel('Month')

# Add y-axis label
ax.set_ylabel('Mole Fraction CO2')

# Add plot title
ax.set_title(fig_title)

# Add legend
ax.legend()

# Setting tick locations
ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

# Change x-axis tick labels to month names
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.show()


#%% Monthly precip calc

# Add columns (month, water_year,day_of_year)
data['month']=data.index.month
data['water_year']=data.index.year
data.loc[data['month']>9,'water_year']= data.loc[data['month']>9,'water_year'] +1
data['day_of_year']=data.index.dayofyear

# Annual data series
data_annual = data.groupby('water_year')['Precip'].sum()

# Calculations
P = np.mean(data_annual) #avg annual total Precip

# Average monthly calc
P_month = P/12

# Creating new CO2 column in new dataframe
data_monthly['Precip'] = np.NAN

for i in range(1,13):
    pmonth = data.loc[data['month']==i,['Precip', 'water_year']]
    monthmean = pmonth.groupby('water_year').sum().mean()
    data_monthly.loc[i,'Precip'] = monthmean[0]
    
    
#%% Precip Bar Chart

fig, ax = plt.subplots()

# Plot monthly precipitation
ax.bar(data_monthly.index, data_monthly['Precip'], label='Monthly CO2')

# Plot average monthly precipitation
ax.axhline(P_month, ls='-', color='k', label='Average Monthly Precipitation (mm/month)')

# Add x-axis label
ax.set_xlabel('Month')

# Add y-axis label
ax.set_ylabel('Monthly Precipitation (mm/month)')

# Add plot title
ax.set_title(fig_title)

# Add legend
ax.legend()

# Setting tick locations
ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

# Change x-axis tick labels to month names
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.show()


#%% Create Polar Plot

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='polar')

# Plot monthly precipitation
theta = np.linspace(0, 2*np.pi, 13)
r = data_monthly['Precip'].tolist()
r.append(r[0])
ax.plot(theta, r)

# Add circular line for average monthly precipitation
avg_precip = np.ones(13) * data_monthly['Precip'].mean()
ax.plot(theta, avg_precip, color='k', linestyle='--')

# Set x-axis (theta) labels to month names
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
ax.set_xticks(theta)
ax.set_xticklabels(month_names)

# Set radial axis (r) ticks
ax.set_yticks(np.arange(0, 110, 20))

# Add labels, legend, and title
ax.set_xlabel('Month')
ax.set_title('Monthly Precipitation at North Orono, ME')
legend = ax.legend(['Monthly Precipitation (mm/month)', 
                    'Average Monthly Precipitation'], loc='lower left')
legend.get_frame().set_linewidth(0)

# Display plot
plt.show()


#%% Plotting Time Series

# Replace LW_OUT values below 100 with NaN
data.loc[data['LW_OUT']<100, 'LW_OUT'] = np.nan

# Replace 0 values with NaN
data.loc[data['SW_IN'] == 0, 'SW_IN'] = np.nan

# Replace 0 values with NaN
data.loc[data['SW_OUT'] == 0, 'SW_OUT'] = np.nan

# Replace 0 values with NaN
data.loc[data['TS'] == 0, 'TS'] = np.nan

# Creating plots
fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1, 
figsize = (10,16), sharex = True)

ax1.plot(data['SW_IN'], 'b-', label = 'SW_IN')
ax1.plot(data['SW_OUT'], 'r-', label = 'SW_OT')
ax1.plot(data['LW_IN'], 'g-', label = 'LW_IN')
ax1.plot(data['LW_OUT'],'c-', label = 'LW_OUT')
ax1.plot(data['LE'], 'k-', label = 'LE')
ax1.plot(data['H'],'m-', label = 'H') 
ax1.legend(loc = 'center left', bbox_to_anchor = (1.0,0.5))

# Add plot components (ax1 - 6)
ax1.set_ylabel('Energy flix in \n watts/m2')
ax1.set_title(fig_title)

ax2.plot(data['WS'],'b-', label = 'Wind Speed')
ax2.set_ylabel('Wind Speed \n m/s')
ax2.legend(loc = 'center left', bbox_to_anchor = (1.0,0.5))


ax3.plot(data['Air_Temp'], 'b-', label = 'Air Temperature')
ax3.plot(data['TS'], 'r-', label = 'Soil Temperatue')
ax3.set_ylabel('Temperature in \n degrees C')
ax3.legend(loc = 'center left', bbox_to_anchor = (1.0,0.5))

# limiting data to only plot before 2013
co2_data = data.loc[data.index.year < 2013, 'CO2']

ax4.plot(co2_data,'b-', label = 'CO2')
ax4.set_ylabel('Mole Fraction CO2')
ax4.legend(loc = 'center left', bbox_to_anchor = (1.0,0.5))


ax5.plot(data['Precip'],'b-', label = 'Precip')
ax5.set_ylabel('Precipitation \n mm')
ax5.legend(loc = 'center left', bbox_to_anchor = (1.0,0.5))

ax6.plot(data['NETRAD'], 'r-', label = 'Net Radiation')
ax6.set_ylabel('Net Radiation in \n watts/m2')
ax6.legend(loc = 'center left', bbox_to_anchor = (1.0,0.5))


#%% Seasonality change calc

# First, convert the daily data into monthly data
df_monthly = data[columns + ['water_year']].resample('M').mean()

# Next, create a new column for the year
df_monthly['year'] = df_monthly.index.year

# Now, group the data by year and calculate the standard deviation for each variable
df_monthly_std = df_monthly.groupby('water_year').std()

# add line to normalize by using average
df_monthly_mean = df_monthly.groupby('water_year').mean()

df_monthly_first = df_monthly_mean.mean()

df_monthly_plot = (df_monthly_std)/(df_monthly_first)


#%% Plot seasonality change

# Set 14 colors using numpy's linspace function
colors = plt.cm.rainbow(np.linspace(0, 1, len(df_monthly_plot.columns)))

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot each variable with its corresponding color
for i, col in enumerate(df_monthly_plot.columns):
    ax.plot(df_monthly_plot.index, df_monthly_plot[col], color=colors[i], label=col)

# Set x-axis label
ax.set_xlabel('Year')

# Set y-axis label
ax.set_ylabel('Standard Deviation')

# Add legend
ax.legend()

# Optional command to make x-tick labels diagonal to avoid overlap
fig.autofmt_xdate()  

# Display the plot
plt.show()

