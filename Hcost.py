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
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')



#Load inputs and output directories, update to your directories
input_fol = "/Users/YOURNAME/Documents/DataRepository/XLFILES/"
output_fol = "/Users/YOURNAME/Documents/DataRepository/XLFILES/Outputs"


#Load Data
data = pd.DataFrame()
data= read_excel(input_fol+"HCOST.xls", sheet_name = "Sheet1", skiprows = 3, header = 0)
data.drop(data.head(0).index, inplace=True)
data = data.iloc[1:].astype(int)


#Descriptive statistics
data.describe().T

# Create correlation matrix
corr_matrix = data.corr().abs()

# Find Correlation for specific columns
data['cost'].corr(data['temp'])

#ScatterPlot for cost and temp + Regression line
data.iplot(
    x='cost',
    y='temp',
    xTitle='Cost',
    yTitle='Heat',
    mode='markers',
    bestfit= True,
    title='Sale Price vs Above ground living area square feet')

#Regression Statistics
model_lin = sm.OLS.from_formula("cost ~ temp", data=data)
result_lin = model_lin.fit()
result_lin.summary()

#Multivariate Regression Statistics
model = sm.GLM.from_formula("Dependent variable ~ independent variables", family=sm.families.Binomial(), data=data)
result = model.fit()
result.summary() 
