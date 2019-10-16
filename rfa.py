import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
dataset.reset_index(drop=True, inplace=True)
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
#print(dataset)


for col in dataset.columns:
    dataset.loc[dataset[col] == -1, col] = np.nan


##Now taking in data to test
test_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
test_data.reset_index(drop=True, inplace=True)

test_data.columns = test_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

for col in test_data.columns:
    test_data.loc[test_data[col] == -1, col] = np.nan

test_data = test_data.drop(columns=['income'])
zeros = [0]*(test_data.shape[0])
test_data['income_in_eur'] = zeros

#print(dataset.shape[0])
#print(test_data.shape[0])

#appending test to rest

data = dataset.append(test_data)
end = data.shape[0]+1
#print(data)



#X_col = ['year_of_record','gender','age','country','size_of_city','profession','university_degree','wears_glasses','body_height_[cm]']
X_col = ['year_of_record','age','size_of_city','gender','country','university_degree','wears_glasses','body_height_[cm]']
y_col = 'income_in_eur'
df_exp = dataset[X_col + [y_col]]

df_exp.dropna(subset=X_col, inplace=True)



X = df_exp[X_col]
y = df_exp.income_in_eur

X.size_of_city = np.log(X.size_of_city)

X = pd.get_dummies(X, columns=['country','university_degree','gender'], drop_first=True)
#X = pd.get_dummies(X, columns=['gender'])
#X = pd.get_dummies(X, columns=['country'])
#X = pd.get_dummies(X, columns=['profession'])

zeros = [0]*(X.shape[0])
X['country_France'] = zeros
X['country_Iceland'] = zeros
X['country_Italy'] = zeros
X['country_Samoa'] = zeros
X['country_Sao Tome & Principe'] = zeros
X['country_Turkey'] = zeros


#print(X.shape[0])

#print('Train')
train_cols = X.columns
#print(X)

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
slice = 96884

X_train = X[0:slice]
X_test = X[slice:end]
y_train = y[0:slice]
y_test = y[slice:end]

#print(X_train)
#print(y_train)

#print(X)
#print("AAAAA",y_train[111990])
#print(y_test)
#print(X_test.year_of_record[73229])
'''

#copy and paste here
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl = mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)
#print(y_pred.shape[0])
#print(y_pred[1])

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


##Now take in test file to print to
X_col = ['year_of_record','age','size_of_city','gender','country','university_degree','wears_glasses','body_height_[cm]']
y_col = 'income_in_eur'
df_exp = test_data[X_col + [y_col]]

df_exp.dropna(subset=X_col, inplace=True)

X = df_exp[X_col]
y = df_exp.income_in_eur

X.size_of_city = np.log(X.size_of_city)

X = pd.get_dummies(X, columns=['country','university_degree','gender'], drop_first=True)


zeros = [0]*(X.shape[0])
test_cols = X.columns

for col in train_cols:
    if col not in test_cols:
        X[col] = zeros

test_cols = X.columns
print(train_cols.shape[0])
print(test_cols.shape[0])

y_pred = mdl.predict(X)
y_pred = np.absolute(y_pred)
print(y_pred)
print(y_pred.shape[0])


over = True
i=0
while(over):
    if test_data.income_in_eur[i] is not 60000:
        test_data.income_in_eur[i] = y_pred[i]
        i = i + 1
        if (i >63353):
            over = False


#print(test_data2)
test_data.to_csv('file4.csv',index=False)


'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)
mdl = LinearRegression()
mdl.fit(X_poly, y_train)
y_pred = mdl.predict(poly_reg.fit_transform(X_test))

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
'''



'''
test_cols = X.columns

zeros = [0]*(X.shape[0])
for col in train_cols:
    if col not in test_cols:
        X[col] = zeros

for col in test_cols:
    if col not in train_cols:
        X.drop(columns = [col])




z = X.drop(columns = ['country_France', 'country_Iceland', 'country_Italy', 'country_Samoa',
       'country_Sao Tome & Principe', 'country_Turkey'])
#print(z)

#print(train_cols)
#print(test_cols)
'''

'''
##Now taking in data to test
test_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
test_data.reset_index(drop=True, inplace=True)

test_data.columns = test_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

for col in test_data.columns:
    test_data.loc[test_data[col] == -1, col] = np.nan

#'year_of_record','age','size_of_city','university_degree','wears_glasses','body_height_[cm]'
#X_col = ['year_of_record','age','size_of_city','university_degree']
y_col = 'income'
df_exp = test_data[X_col + [y_col]]

df_exp.dropna(subset=X_col, inplace=True)

X2 = df_exp[X_col]
y2 = df_exp.income

X2.size_of_city = np.log(X2.size_of_city)

X2 = pd.get_dummies(X2, columns=['country','university_degree','gender'], drop_first=True)


'''

'''
y_pred = mdl.predict(X)
X['income'] = y_pred


##Now take in test file to print to
test_data2 = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
test_data2.reset_index(drop=True, inplace=True)
test_data2.columns = test_data2.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

#print(test_data2)

for i in range(73230):
    try:
        test_data2.income[i] = X.income[i]
    except:
        test_data2.income[i] = 60000



test_data2.to_csv('file3.csv',index=False)
'''







'''
***Not very accurate***
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl = mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
'''

'''
***Takes very long and is not very accurate***
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
'''

'''
***Does not work for data other than that it was trained on***
from sklearn.tree import DecisionTreeRegressor
mdl = DecisionTreeRegressor()
mdl = mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_train)

from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
'''