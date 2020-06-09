# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:03:11 2020

@author: user
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import pickle

#read the data

nutrition_df = pd.read_csv('Book.csv', header=0)
df = nutrition_df.drop(["FoodID", "FoodName","Brand","Category","Swimming","Cycling","Jogging","Walking"], axis=1)
eg=nutrition_df[nutrition_df['FoodID'] == 5]

#1.Bars, Breakfast Cereals
#2.Beverages (Sports, Energy & Meal Shakes, Soda, Coffee, Tea)
#3.Breads, Bagels, Rolls, Crackers, Cookies
#4.Cakes, Muffins, Donuts, Baking
#5.Candy, Chocolate, Cough Drops, Supplements
#6.Cheese, Cream, Ice Cream & Yogurts
#7.Desserts, Pancakes, Pastries, Pies
#8.Eggs, Meats, Poultry, Seafoods
#9.Fast-Foods, Fair Foods, Eating Out, Restaurants
#10.Fats, Condiments, Sauces, Dressings
#11.Frozen & Packaged Meals & Pizzas, Soup, Tofu
#12.Grains & Flour, Rice & Pasta
#13.Snacks: Chips, Popcorn, Pretzels
#14.Sugar, Honey, Syrups, Toppings
#15.Alcoholic Drinks

#categorical daa
#covert the column into float datatype
df['Weight'] = pd.to_numeric(df['Weight'],errors='coerce')
df['Sodium'] = pd.to_numeric(df['Sodium'],errors='coerce')
df['Calories'] = pd.to_numeric(df['Calories'],errors='coerce')
df['Fat'] = pd.to_numeric(df['Fat'],errors='coerce')
df['Calcium'] = pd.to_numeric(df['Calcium'],errors='coerce')
df['Potassium'] = pd.to_numeric(df['Potassium'],errors='coerce')
df['Iron'] = pd.to_numeric(df['Iron'],errors='coerce')
df['VitaminA'] = pd.to_numeric(df['VitaminA'],errors='coerce')
df['VitaminC'] = pd.to_numeric(df['VitaminC'],errors='coerce')


#replace the nan val to 0 , we cann't substitute it with the mean or median or any other value
df = df.replace(np.nan, 0, regex=True)
df.to_csv('data.csv',index=False)

k= df.describe()
#finding the top ten nearest neighbours
nbrs = NearestNeighbors(n_neighbors=5,algorithm='ball_tree',metric='euclidean').fit(df)
joblib.dump(nbrs, df)

pickle.dump(nbrs, open('model.pkl', 'wb'))



#distances, indices = nbrs.kneighbors([[48.2, 180, 4.5, 150, 28, 0,10,0.22,0.26,0.009,3,0.036]], n_neighbors=10)
#recommended_products = [nutrition_df.loc[i,"FoodName"] for i in indices[0]]


#print (recommended_products)



"""
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values="NaN",strategy='mean',axis=0) 
imputer=imputer.fit(X[:,2:11])
X[:,11]=imputer.transform(X[:,11])
"""

"""
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder=LabelEncoder()
test_data[:,0]=labelencoder.fit_transform(test_data[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
test_data=onehotencoder.fit_transform(test_data).toarray()
"""

"""
#dataset=dataset.drop('Alcohol',axis=1)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values
"""
