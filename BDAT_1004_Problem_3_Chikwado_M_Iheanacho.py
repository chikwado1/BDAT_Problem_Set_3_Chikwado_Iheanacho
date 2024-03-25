#!/usr/bin/env python
# coding: utf-8

# # QUESTION 1
# Introduction:
# Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.
# Occupations
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user.
# Step 3. Assign it to a variable called users
# Step 4. Discover what is the mean age per occupation
# Step 5. Discover the Male ratio per occupation and sort it from the most to the least
# Step 6. For each occupation, calculate the minimum and maximum ages
# Step 7. For each combination of occupation and sex, calculate the mean age
# Step 8. For each occupation present the percentage of women and men
# 

# In[1]:


# Step 1. Import the necessary libraries
import pandas as pd

# Step 2. Import the dataset
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
users = pd.read_csv(url, sep='|')
users.head()
users.tail()
# Step 3. Assign it to a variable called users


# In[7]:


# Step 4. the mean age per occupation
mean_age_per_occupation = users.groupby('occupation')['age'].mean()
print("Mean age per occupation:\n", mean_age_per_occupation)


# In[8]:


# Step 5. Discover the Male ratio per occupation and sort it from the most to the least
male_ratio_per_occupation = users.pivot_table(index='occupation', columns='gender', aggfunc='size', fill_value=0)
male_ratio_per_occupation['male_ratio'] = male_ratio_per_occupation['M'] / (male_ratio_per_occupation['M'] + male_ratio_per_occupation['F'])
male_ratio_per_occupation_sorted = male_ratio_per_occupation['male_ratio'].sort_values(ascending=False)
print("\nMale ratio per occupation (sorted):\n", male_ratio_per_occupation_sorted)


# In[12]:


# Step 6. For each occupation, calculate the minimum and maximum ages
min_max_age_per_occupation = users.groupby('occupation')['age'].agg(['min', 'max'])
print("\nMinimum and maximum ages per occupation:\n", min_max_age_per_occupation)


# In[32]:


# Step 7. For each combination and sex, calculate the mean age
users.groupby(['occupation', 'gender']).age.mean()


# In[33]:


# Step 8. For each occupation present the percentage of women and men
occupation_gender_counts = users.groupby('occupation')['gender'].value_counts()
occupation_gender_percentage = occupation_gender_counts.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
print("\nPercentage of women and men per occupation:\n", occupation_gender_percentage)


# In[ ]:





# In[ ]:





# # QUESTION 2
# #Euro Teams
# #Step 1. Import the necessary libraries
# #Step 2. Import the dataset from this https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv
# #Step 3. Assign it to a variable called euro12
# #Step 4. Select only the Goal column
# #Step 5. How many team participated in the Euro2012?
# #Step 6. What is the number of columns in the dataset?
# #Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline
# #Step 8. Sort the teams by Red Cards, then to Yellow Cards
# #Step 9. Calculate the mean Yellow Cards given per Team
# #Step 10. Filter teams that scored more than 6 goals
# #Step 11. Select the teams that start with G
# #Step 12. Select the first 7 columnS
# #Step 13. Select all columns except the last 3
# #Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# In[17]:


#import the necessary libraries
import pandas as pd


# In[20]:


#import dataset and assigned it to variable euro12
euro12 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')
euro12.head()


# In[21]:


#select Goal Column
euro12[['Goals']]


# In[22]:


#count of teams participated
euro12['Team'].count()


# In[23]:


#Number of columns in dataset
euro12.shape[1]


# In[24]:


#View only the columns Team, yellow cards and Red cards and assign them to a dataframe called discipline
discipline = pd.DataFrame(euro12, columns=['Team', 'Yellow Cards', 'Red Cards'])
discipline


# In[25]:


#Sort the teams by Red Cards, then to Yellow Cards
discipline.sort_values(by = ['Red Cards','Yellow Cards'], inplace = True)
discipline


# In[26]:


#Calculate the mean Yellow Cards given per Team
euro12.groupby('Team')['Yellow Cards'].mean()


# In[27]:


#Filter teams that scored more than six(6) goals
euro12[euro12['Goals'] > 6]


# In[28]:


#Select the teams that start with G
euro12[euro12.Team.str.startswith('G')]


# In[29]:


#Select first seven(7) columns
euro12.iloc[ :, : 7]


# In[30]:


#Select all columns except the last 3
euro12.iloc[ : , : -3]


# In[31]:


#Present only the shooting accuracy from England, Italy and Russia
euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']),['Team','Shooting Accuracy']]


# In[ ]:





# # QUESTION 3
# #Housing
# #Step 1. Import the necessary libraries
# #Step 2. Create 3 differents Series, each of length 100, as follows:
# • The first a random number from 1 to 4
# • The second a random number from 1 to 3
# • The third a random number from 10,000 to 30,000
# #Step 3. Create a DataFrame by joinning the Series by column
# #Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
# #Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'
# #Step 6. Ops it seems it is going only until index 99. Is it true?
# #Step 7. Reindex the DataFrame so it goes from 0 to 299

# In[35]:


#import libraries
import pandas as pd
import random


# In[37]:


#create dataframes as described by joining the series
first = [[random.randint(1,4)] for i in range(100)]
df1 = pd.DataFrame(first)

second = [[random.randint(1, 3)] for i in range(100)]
df2 = pd.DataFrame(second)

third = [[random.randint(10000, 30000)] for i in range(100)]
df3 = pd.DataFrame(third)

df = pd.concat([df1, df2, df3], axis=1)
df


# In[38]:


#change the column names
df.columns = ['bedrs', 'bathrs', 'price_sqr_meter']
df


# In[39]:


#Create a one column DataFrame with the values of the three (3) series and assign it to 'bigcolumn'
df_new = df.bedrs.astype(str).str.cat(df.bathrs.astype(str)).str.cat(df.price_sqr_meter.astype(str))
df_new.columns = ['bigcolumn']
df_new


# In[ ]:





# In[ ]:





# # QUESTION 4
# #Wind Statistics
# #The data have been modified to contain some missing values, identified by NaN.
# #Using pandas should make this exercise easier, in particular for the bonus question.
# #You should be able to perform all of these operations without using a for loop or other looping construct.
# #The data in 'wind.data' has the following format:
# Yr Mo Dy RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BEL MAL
# 61 1 1 15.04 14.96 13.17 9.29 NaN 9.87 13.67 10.25 10.83 12.58 18.50 15.04
# 61 1 2 14.71 NaN 10.83 6.50 12.62 7.67 11.50 10.04 9.79 9.67 17.54 13.83
# 61 1 3 18.50 16.88 12.33 10.13 11.17 6.17 11.25 NaN 8.50 7.67 12.75 12.71
# The first three columns are year, month, and day. The remaining 12 columns are average windspeeds in knots at 12 locations in Ireland on that day.
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from the attached file wind.txt
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index.
# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.
# Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].
# Step 6. Compute how many values are missing for each location over the entire record.They should be ignored in all calculations below.
# Step 7. Compute how many non-missing values there are in total.
# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
# A single number for the entire dataset.
# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days
# A different set of numbers for each location.
# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.
# A different set of numbers for each day.
# Step 11. Find the average windspeed in January for each location.
# Treat January 1961 and January 1962 both as January.
# Step 12. Downsample the record to a yearly frequency for each location.
# Step 13. Downsample the record to a monthly frequency for each location.
# Step 14. Downsample the record to a weekly frequency for each location.
# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.

# In[48]:


# Import the necessary libraries
import pandas as pd
import numpy as np
# Import the dataset
data = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data', sep = '\s+')
data.head()


# In[ ]:





# In[49]:


#Assign it to a variable called data and replace the first 3 columns by a proper datetime index
data["Date"] = pd.to_datetime(data[["Yr","Mo","Dy"]].astype(str).agg('-'.join,axis=1))
data = data.drop(columns = ["Yr","Mo","Dy"])
data = data[['Date', 'RPT', 'VAL', 'ROS', 'KIL', 'SHA', 'BIR', 'DUB', 'CLA', 'MUL', 'CLO', 'BEL', 'MAL']]
data.head()


# In[50]:


#Year 2061? Do we really have data from this year? Create a function to fix it and apply it
NData = data.set_index("Date")
#Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].
NData.index.astype("datetime64[ns]")


# In[51]:


# . Compute how many values are missing for each location over the entire record.They should be ignored in all calculations below
NData.isnull().values.ravel().sum()


# In[52]:


#. Compute how many non-missing values there are in total.
nData = NData.count()
print("Total non-missing values are :", nData.sum())


# In[53]:


#Calculate the mean windspeeds of the windspeeds over all the locations and all the times. A single number for the entire dataset.
x = round(NData.mean(), 2)
x


# In[54]:


#Single number for the entire dataset
round(x.mean(),2)


# In[56]:


#Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days A different set of numbers for each location.
#Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day. A different set of numbers for each day.
def statsKPI(x):
    x = pd.Series(x)
    Mean = x.mean()
    Std = x.std()
    Min = x.min()
    Max = x.max()
    res = [Min,Max,Mean,Std]
    indx = ["MEAN", "Std", "Min", "Max"]
    result = pd.Series(res,index = indx)
    return result

#A different set of numbers for each location
loc_stats = NData.apply(statsKPI)
loc_stats


# In[59]:


#Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at eachday.A different set of numbers for each day
day_stats = NData.apply(statsKPI, axis = 1)
day_stats.head()


# In[60]:


#Find the average windspeed in January for each location. Treat January 1961 and January 1962 both as January
jan_AVG_wspeed = NData[NData.index.month ==1]
round(jan_AVG_wspeed.mean(),2)


# In[61]:


#Downsample the record to a yearly frequency for each location.
print( "Yearly:\n", NData.resample('A').mean())


# In[62]:


#Downsample the record to a monthly frequency for each location. 
print("Monthly:\n", NData.resample('M').mean())


# In[63]:


#Downsample the record to a weekly frequency for each location.
print("Weekly:\n", NData.resample('W').mean())


# In[64]:


#Calculate the min, max and mean windspeeds and standard deviations of thewindspeeds across all locations for each week (assume that the first week starts on BJanuary 2 1961) for the first 52 weeks.
stats1 = NData.resample('W').mean().apply(lambda x: [x.count(),x.min(),x.max(),x.mean(),x.std()])
print(stats1)


# In[65]:


first_year = NData[NData.index.year == 1961]
stats1 = NData.resample('W').mean().apply(lambda x: x.describe())
print(stats1)


# In[ ]:





# In[ ]:





# In[ ]:





# # QUESTION 5
# Question 5
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv.
# Step 3. Assign it to a variable called chipo.
# Step 4. See the first 10 entries
# Step 5. What is the number of observations in the dataset?
# Step 6. What is the number of columns in the dataset?
# Step 7. Print the name of all the columns.
# Step 8. How is the dataset indexed?
# Step 9. Which was the most-ordered item?
# Step 10. For the most-ordered item, how many items were ordered?
# Step 11. What was the most ordered item in the choice_description column?
# Step 12. How many items were orderd in total?
# Step 13.
# • Turn the item price into a float
# • Check the item price type
# • Create a lambda function and change the type of item price
# • Check the item price type
# Step 14. How much was the revenue for the period in the dataset?
# Step 15. How many orders were made in the period?
# Step 16. What is the average revenue amount per order?
# Step 17. How many different items are sold?

# In[66]:


#Import the necessary libraries
import pandas as pd
import numpy as np

#. Import the dataset from this address.Step 3. Assign it to a variable called chipo.
chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep ='\t')

#See the first 10 entries
chipo.head(10)


# In[67]:


#What is the number of observations in the dataset?
chipo.info()


# In[70]:


#. What is the number of columns in the dataset?
chipo.shape[1]



# In[71]:


#Print the name of all the columns. 
chipo.columns


# In[72]:


#How is dataset indexed?
#index check
chipo.index


# In[73]:


#which was the most ordered item?
item = chipo.groupby('item_name')
item = item.sum()
item = item.sort_values(['quantity'], ascending = False)
item.head(1)


# In[74]:


#What was the most ordered item in the choice_description column? 
chipo.groupby('choice_description').sum().sort_values(['quantity'], ascending = False).head(1)


# In[77]:


#How many items were orderd in total?
chipo.quantity.sum()


# In[78]:


#Check the item price type
chipo.item_price.dtype


# In[ ]:


#Create a lambda function and change the type of item price
chgpriceType = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(chgpriceType)


# In[82]:


chipo.item_price.dtype


# In[84]:


#. How much was the revenue for the period in the dataset? 
revenue = (chipo['quantity']* chipo['item_price']).sum()
print('Revenue was: $' + str(np.round(revenue, 2)))


# In[85]:


#How many orders were made in the period?
chipo.order_id.value_counts().count()


# In[ ]:





# In[94]:


#How many different items are sold?
chipo.item_name.value_counts().count()


# In[ ]:





# # QUESTION 6
# Create a line plot showing the number of marriages and divorces per capita in the U.S. between 1867 and 2014. 
# Label both lines and show the legend.
# Don't forget to label your axes!

# In[99]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/chikw/Downloads/us-marriages-divorces-1867-2014.csv"
df = pd.read_csv(file_path)

# Filter data for years between 1867 and 2014
df_filtered = df[(df['Year'] >= 1867) & (df['Year'] <= 2014)]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['Year'], df_filtered['Marriages_per_1000'], label='Marriages per 1000')
plt.plot(df_filtered['Year'], df_filtered['Divorces_per_1000'], label='Divorces per 1000')

# Add title and labels
plt.title('Number of Marriages and Divorces per Capita in the U.S. (1867-2014)')
plt.xlabel('Year')
plt.ylabel('Number per Capita')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # QUESTION 7
# Create a vertical bar chart comparing the number of marriages and divorces per capita in the U.S. between 1900, 1950, and 2000.
# Don't forget to label your axes!

# In[111]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/chikw/Downloads/us-marriages-divorces-1867-2014.csv"
df = pd.read_csv(file_path)

# Filter data for years 1900, 1950, and 2000
years_of_interest = [1900, 1950, 2000]
df_filtered = df[df['Year'].isin(years_of_interest)]


# In[100]:


# Plot the data
plt.figure(figsize=(10, 6))
plt.bar(df_filtered['Year'] - 0.2, df_filtered['Marriages_per_1000'], width=0.4, label='Marriages per 1000')
plt.bar(df_filtered['Year'] + 0.2, df_filtered['Divorces_per_1000'], width=0.4, label='Divorces per 1000')

# Add title and labels
plt.title('Number of Marriages and Divorces per Capita in the U.S. (1900, 1950, 2000)')
plt.xlabel('Year')
plt.ylabel('Number per Capita')

# Add legend
plt.legend()

# Show plot
plt.xticks(years_of_interest)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # QUESTION 8
# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort the actors by their kill count and label each bar with the corresponding actor's name. Don't forget to label your axes!

# In[108]:


data = pd.read_csv('C:/Users/chikw/Downloads/actor_kill_counts.csv')
newData = data.sort_values("Count", ascending = True)
newData


# In[110]:


newData.plot.barh(x='Actor', y='Count', color = '#FFAE39')
plt.title('Actors kill Count', fontsize = 18, color = 'Green')
plt.xlabel('Count of kills', color = 'Black', fontsize = 15)
plt.ylabel('Actor', color = 'Black', fontsize = 15)


# In[ ]:





# # QUESTION 9
# Create a pie chart showing the fraction of all Roman Emperors that were assassinated.
# Make sure that the pie chart is an even circle, labels the categories, and shows the percentage breakdown of the categories.

# In[132]:


import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('C:/Users/chikw/Downloads/roman-emperor-reigns.csv')
data.head()


# In[133]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

newdata = data.groupby("Cause_of_Death")['Emperor'].count()
newdata


# In[134]:


newdata.plot.pie(autopct="%.1f%%", figsize = (8,8), fontsize = 20, ylabel = '', explode = [0.01]*8)
plt.title("Roman Emperor's Cause of Death", color = 'Black', fontsize = 18, fontweight = 'bold')


# In[ ]:





# # QUESTION 10
# Create a scatter plot showing the relationship between the total revenue earned by arcades and the number of Computer Science PhDs awarded in the U.S. between 2000 and 2009.
# Don't forget to label your axes!
# Color each dot according to its year.

# In[135]:


import pandas as pd

# Load the data from the CSV file
data = pd.read_csv("C:/Users/chikw/Downloads/arcade-revenue-vs-cs-doctorates.csv")
data.head()


# In[137]:


import matplotlib.pyplot as pit
get_ipython().run_line_magic('matplotlib', 'inline')

colours = ['#C4E57D', '#2AC280', '#FFAE39', '#DC5B3B', '#52E397', '#C5CBA3', '#9CD5F6', '#6E50D9', '#9A5E59', '#9BC8F5']

data.plot.scatter(x = 'Total Arcade Revenue (billions)',
                 y = 'Computer Science Doctorates Awarded (US)', c = colours, s =50, figsize = (5,4))

plt.title('Revenue Vs CS Doctorates', color = 'Black', fontsize = 18)
plt.xlabel('Total Arcade Revenue (billions)', color = 'Black', fontsize = 12)
plt.ylabel('CS Doctorates Awarded', color = 'Black', fontsize = 12)


# In[138]:


data.plot.scatter(x = 'Total Arcade Revenue (billions)',
                 y = 'Computer Science Doctorates Awarded (US)', c = 'Year', s = 50)

plt.title('Revenue Vs CS Doctorates', color = 'Black', fontsize = 18)
plt.xlabel('Total Arcade Revenue (billions)', color = 'Black', fontsize = 12)
plt.ylabel('CS Doctorates Awarded', color = 'Black', fontsize = 12)


# In[ ]:





# In[ ]:





# In[ ]:




