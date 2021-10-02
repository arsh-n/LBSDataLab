#Import directories
import os
import pandas as pd
from pandas import read_excel
import numpy as np
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly
from plotly import tools
import matplotlib
import cufflinks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#Load inputs and output directories 
input_fol = "/Users/arshnagpal/Documents/DataRepository/XLFILES/"
output_fol = "/Users/arshnagpal/Documents/DataRepository/XLFILES/Outputs"

#Load Data
data = pd.DataFrame()
data= read_excel(input_fol+"HousePriceData2006.xlsx", sheet_name = "HousePriceData2006", header = 0)
data.drop(data.head(0).index, inplace=True)
data = data.iloc[1:]

#price outliers
data = data[data.price != 1650000]

#lotSize outliers 
XlotSize = (0,12.2, 8.35, 8.97, 7.24)
data = data[-data['lotSize'].isin(XlotSize)] 

#livingArea outliers
data = data[data.livingArea != 14540]
data = data[data.livingArea != 35.11]

#Descriptive statistics
data.describe().T

#creating dummy variables

data['newConstruction_Yes'] = pd.get_dummies(data=data['newConstruction'], drop_first=True)
data['centralAir_Yes'] = pd.get_dummies(data=data['centralAir'], drop_first=True)
data['waterfront_Yes'] = pd.get_dummies(data=data['waterfront'], drop_first=True)

#ScatterPlot for cost and temp + Regression line
data.iplot(
    x='price',
    y='landValue',
    xTitle='price',
    yTitle='landValue',
    mode='markers',
    bestfit= True,
    title='Price vs landValue')
    
#Equation for linear regression trend line for one variable lotsize
X = data['price'].values.reshape(-1,1)
y = data['lotSize'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

#test LR for one variable
X = data['price']
y = data['landValue']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

#Equation for linear regression trend line for all variables minus the categorical variables
Xs = data.drop(['price', 'heating', 'fuel', 'sewer', 'waterfront', 'newConstruction', 'centralAir'], axis=1)
y = data['price'].values.reshape(-1,1)
reg = LinearRegression()
reg2 = reg.fit(Xs, y)
print("The linear model is: Y = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))

#test LR for all variables to make final model
X = np.column_stack((data['age'], data['lotSize'], data['landValue'], 
                     data['livingArea'], #data['pctCollege'], 
                     data['bedrooms'], 
                     data['bathrooms'], 
                     data['rooms'],data['waterfront_Yes'], data['newConstruction_Yes'],data['centralAir_Yes']))
y = data['price']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary(xname=['price','age', 'lotSize', 'landValue', 
                     'livingArea', #'pctCollege', 
                     'bedrooms', 

                    'bathrooms', 
                     'rooms','waterfront', 'newConstruction','centralAir']))#'heating','fuel', 'sewer',

#testing one residual plot
x = data['price']
y = data['landValue']

sns.residplot(x,y)

# Find Correlation for specific columns
data['price'].corr(data['lotSize'])



