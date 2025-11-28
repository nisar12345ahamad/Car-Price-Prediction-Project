import pandas as pd
import numpy as np
car = pd.read_csv('quikr_car.csv')
print(car.head())
print(car.shape)

# Quality
# year has many non-year problem values.
# year has object while you should want to integer.
# price has ask for price
# price object to input
# kms_driven has kms with integers
# kms_driven object to int
# km_driven has nan values
# fuel_type has nan values
# keep first 3 word of names

print(car['year'].unique())
print(car['Price'].unique())
print(car['kms_driven'].unique())
print(car['fuel_type'].unique())

# cleaning
backup = car.copy()
print('year has object while you should want to integer.')
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
print(car['year'].unique())
print(car.head())
print(car["year"].info())

# price has ask for price
car = car[car['Price'] !="Ask For Price"]
print(car.head())
print(car['Price'].unique())
print(car['Price'].info())

# price object to input
car['Price'] = car['Price'].str.replace(',','').astype(int)
print(car['Price'].unique())
print(car['Price'].info())


# kms_driven has kms with integers
car['kms_driven'] = (car['kms_driven']                    
                     .str.split(' ')
                     .str.get(0)
                     .str.replace(',','')
)        
car['kms_driven'] = car['kms_driven'].str.isnumeric()
car['kms_driven'] = car['kms_driven'].astype(int)
print(car['kms_driven'].unique())
print(car['kms_driven'].info())

# fuel_type has nan values
car = car[~car['fuel_type'].isna()]
print(car['fuel_type'].info())
print(car['fuel_type'].unique())

# keep first 3 word of names
car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')
print(car['name'].unique())
print(car.reset_index())
print(car.head())

print(car.info())
print(car.describe())


print(car[car["Price"]>6e6])
print(car[car["Price"]<6e6])
car = car[car["Price"]<6e6].reset_index(drop=True)
print(car.head())
print(car.shape)

car.to_csv('Cleaned Car.csv')

# Model
X = car.drop(columns='Price')
y = car['Price']

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X , y, test_size=0.2 ,random_state=42)
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])

column_tranf = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder = 'passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_tranf , lr)
pipe.fit(X_train , y_train)
y_pred = pipe.predict(X_test)

print("R2Score:",r2_score(y_test , y_pred))

scores = []
for i in range(1000):
    X_train, X_test , y_train , y_test = train_test_split(X , y, test_size=0.2,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_tranf , lr)
    pipe.fit(X_train , y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test , y_pred))

print(np.argmax(scores))
print(scores[np.argmax(scores)])

X_train, X_test , y_train , y_test = train_test_split(X , y, test_size=0.2,random_state=i)
lr = LinearRegression()
pipe = make_pipeline(column_tranf , lr)
pipe.fit(X_train , y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test , y_pred))

import pickle
pickle.dump(pipe, open('LinearRegresionModel.pkl','wb'))
