import pandas as pd
import xlrd
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Import file
df = pd.read_excel(r'C:\Users\Konstantinos\Downloads\Port_of_Twente\VesselData.xlsx')

# Single imputation method to fill missing values in a column with the most frequent
imp = SimpleImputer(strategy="most_frequent")
imp.fit_transform(df)

# Remove null values
df[df.isnull().any(axis=1)]

# Drop all date fields. They can also be kept though, if we want to check additional variables (e.g. arrival delay in dates)
df=df[['vesseldwt','vesseltype','discharge1','discharge2','discharge3','discharge4','load1','load2','load3','load4','traveltype','isremarkable','previousportid','nextportid','vesselid']]

# Convert categorical variables to categories and apply encoding to convert to numerical
df["traveltype"] = df["traveltype"].astype('category')
df["isremarkable"] = df["isremarkable"].astype('category')
df["traveltype"] = df["traveltype"].cat.codes
df["isremarkable"] = df["isremarkable"].cat.codes

# Apply label encoding to some numerical variables, since their values are arbitrary (e.g. higher port number doesn't mean necessarily "greater")
label_encoder = LabelEncoder()
df['vesseltype'] = label_encoder.fit_transform(df['vesseltype'])
df['previousportid'] = label_encoder.fit_transform(df['previousportid'])
df['nextportid'] = label_encoder.fit_transform(df['nextportid'])
df['vesselid'] = label_encoder.fit_transform(df['vesselid'])

#print(df.head())

# Drop rows with nan values
df = df.dropna()

# Here we provide an example by building a Random Forest regression model to predict the
# discharge for cargo type 1. Due to time limitations, I only created 2 out of the 8 models
# (for the 8 different target variables). Alternatively however we can also create a multiple
# output regression model, to predict all 8 variables using a single model

# Set target variable (discharge 1), and remove it from the dataset of predictors
labels = np.array(df['discharge1'])
df2= df.drop('discharge1', axis = 1)

# We keep as predictors only the variables not related to discharge or load, since each
# time discharge and load will be unknown and have to be predicted
df2=df2[['traveltype','previousportid','nextportid','isremarkable','vesselid','vesseldwt']]

# Saving feature names for later use
feature_list = list(df2.columns)
# Convert to numpy array
features = np.array(df2)

# Split into train and test data randomly, using 80:20 ratio
train_features, test_features, train_labels, test_labels = train_test_split(df2, labels, test_size = 0.2, random_state = 42)

# Create a Random Forest of 1000 trees and train the model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 10)
rf.fit(train_features, train_labels)

# Make predictions on test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

print('Mean Absolute Error for discharge1:', round(np.mean(errors), 2), 'discharge activities.')


# Set target variable (load 1), and remove it from the dataset of predictors
labels = np.array(df['load1'])
df= df.drop('load1', axis = 1)

# We keep as predictors only the variables not related to discharge or load, since each
# time discharge and load will be unknown and have to be predicted
df3=df[['traveltype','previousportid','nextportid','isremarkable','vesselid','vesseldwt']]

# Saving feature names for later use
feature_list = list(df3.columns)
# Convert to numpy array
features = np.array(df)

# Split into train and test data randomly, using 80:20 ratio
train_features, test_features, train_labels, test_labels = train_test_split(df3, labels, test_size = 0.2, random_state = 42)

# Create a Random Forest of 1000 trees and train the model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 10)
rf.fit(train_features, train_labels)

# Make predictions on test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

print('Mean Absolute Error for load1:', round(np.mean(errors), 2), 'kgs.')
