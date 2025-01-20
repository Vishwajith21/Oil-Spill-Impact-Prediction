import numpy as np
import matplotlib.pyplot as plt

import pandas as pd





# Load the iris dataset
colums = ['spill_size','toxicity_level','distance_to_coast','impact']
df = pd.read_csv('oil_spill_synthetic_data.csv',names=colums)
df.head()
df['spill_size'] = pd.to_numeric(df['spill_size'], errors='coerce') # Use errors='coerce' to handle invalid values
df['toxicity_level'] = pd.to_numeric(df['toxicity_level'], errors='coerce') # Use errors='coerce' to handle invalid values
df['distance_to_coast'] = pd.to_numeric(df['distance_to_coast'], errors='coerce')


     

# visualization of data
df.describe()

#separate the input and output
data= df.values
X=data[:,0:3]
Y=data[:,3] # Access the last column using index 3
print(X)
print(Y)
#let train the model
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(X_test.shape)
# Import the SimpleImputer
from sklearn.impute import SimpleImputer

# Create an imputer to replace NaN with the mean
imputer = SimpleImputer(strategy='mean')  # You can use 'median' or 'most_frequent' as well

# Fit the imputer to your training data and transform it
X_train = imputer.fit_transform(X_train)

# Transform the test data using the trained imputer
X_test = imputer.transform(X_test)

# Now you can train your model
model_svc = SVC()
model_svc.fit(X_train, Y_train)

#support vector machine learning algorithm
from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(X_train,Y_train)
prediction = model_svc.predict(X_test)
#check the prediction accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction))
#checking the model by giving manual value
test_model = model_svc.predict([[5.3,2.5,4.6]])
print("The Impact level is :{}".format(test_model))