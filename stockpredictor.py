# Multiply Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt, mpld3

import pandas as pd
import shutil, os;




List = open("C:\\Users\\MLH\\Downloads\\info.txt").read().split()

my_1 = List[0] = float(List[0])
my_2 = List[1] = float(List[1])

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,6].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination (significance level = 5)
import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) # create a constant in the regression model

X_opt = X[:,[0, 1, 2, 3, 4]] # creates a new set for optimal X values
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)

X_opt = X[:,[0, 1, 3, 4]] # creates a new set for optimal X values
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)

X_opt = X[:,[1, 3, 4]] # creates a new set for optimal X values
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)

X_opt = X[:,[1, 4]] # creates a new set for optimal X values
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)


###################################################################################################


X_opt2 = dataset.iloc[:, 1].values
y_opt = dataset.iloc[:, 6].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_opt2, y_opt, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
X_train2 = X_train2.reshape(-1, 1)
y_train2 = y_train2.reshape(-1, 1)
regressor2 = LinearRegression()
regressor2.fit(X_train2, y_train2)

# Predicting the test set
X_test2 = X_test2.reshape(-1, 1)
y_pred2 = regressor2.predict(X_test2)

# Visualizing the training set details

x_var = X_train2.tolist()
y_var = y_train2.tolist()
x_var2 = [x[0] for x in x_var]
y_var2 = [y[0] for y in y_var]

fig = plt.figure()
#plt.scatter(x_var2, y_var2, color = 'red') # graph the real set of observations in graph
fid = plt.plot(x_var2,regressor2.predict(x_var), color = 'blue') # graph the line of best fit
plt.title('Stock Predictor')
plt.xlabel('Revenue')
plt.ylabel('Profit')
mpld3.save_html(fig,"revenue_plot.html")
mpld3.fig_to_html(fig,template_type="simple")
#mpld3.disable_notebook()
plt.show() # specify that this is end of graph


# Visualizing the test set details

plt.scatter(X_test2, y_test2, color = 'red') # graph the test set of observations in graph
#plt.plot(x_var2,regressor2.predict(x_var2), color = 'blue' ) # graph the line of best fit
plt.title('Stock Predictor')
plt.xlabel('Revenue')
plt.ylabel('Profit')
#mpld3.show()
plt.savefig('revenue_scatter.png')

plt.show() # specify that this is end of graph
y_predf = regressor2.predict(my_1) 

########################################################################################################


X_opt3 = dataset.iloc[:, 5:6].values
y_opt2 = dataset.iloc[:, 6:7].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_opt3, y_opt2, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor3 = LinearRegression()
regressor3.fit(X_train3, y_train3)

# Predicting the test set
y_pred2 = regressor3.predict(X_opt3)

x_vari = X_train3.tolist()
y_vari = y_train3.tolist()
x_vari2 = [x[0] for x in x_var]
y_vari2 = [y[0] for y in y_var]


# Visualizing the training set details
#plt.scatter(X_train3, y_train3, color = 'red') # graph the real set of observations in graph

fig2 = plt.figure()

fid2 = plt.plot(x_vari,regressor3.predict(x_vari), color = 'blue' ) # graph the line of best fit
plt.title('Stock Predictor')
plt.xlabel('Shareholders Equity')
plt.ylabel('Profit')
mpld3.save_html(fig2,"holder_plot.html")
mpld3.fig_to_html(fig,template_type="simple")
#mpld3.show()
plt.show() # specify that this is end of graph


# Visualizing the test set details
plt.scatter(X_test3, y_test3, color = 'red') # graph the test set of observations in graph
#plt.plot(X_train3,regressor3.predict(X_train3), color = 'blue' ) # graph the line of best fit
plt.title('Stock Predictor')
plt.xlabel('Shareholders Equity')
plt.ylabel('Profit')
plt.savefig('holder_scatter.png')
plt.show() # specify that this is end of graph

y_predf2 = regressor3.predict(my_2) 

#############################################################################################################

final_predict = (y_predf+y_predf2)/2
txt = final_predict.flat[0]
txt = round(txt, 2)
txt = txt.astype('str')
with open("info.txt", "w") as text_file:
    text_file.write(txt)

print("The final Profit of the company")
print("\t ↓")

print(final_predict[0])
##############################################################################################################


