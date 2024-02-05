from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import statsmodels
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

dc = pd.read_csv('Auto.csv')
scatter_matrix(dc)
#plt.show()

#x = dc.loc[:, dc.columns != 'name']

dc.drop(columns=['name'])
dc = dc[dc.horsepower != '?']
dc['horsepower'] = dc['horsepower'].astype('int64')
#print(dc)
y = dc.corr(numeric_only=True)
#print(y)

plt.figure(figsize=(10,8),linewidth=10,edgecolor="#04253a" )
sns.heatmap(y, annot=True, cmap=plt.cm.Reds)
#plt.show()

#statsmodels.graphics.regressionplots.influence_plot(dc)
#plt.show()


results = smf.ols('mpg ~ cylinders', data=dc).fit()
sm.graphics.influence_plot(results)
plt.title("cylinders")
#plt.show()

results = smf.ols('mpg ~ displacement', data=dc).fit()
sm.graphics.influence_plot(results)
plt.title("displacement")
#plt.show()

results = smf.ols('mpg ~ horsepower', data=dc).fit()
sm.graphics.influence_plot(results)
plt.title("horsepower")
#plt.show()

results = smf.ols('mpg ~ weight', data=dc).fit()
sm.graphics.influence_plot(results)
plt.title("weight")
#plt.show()

results = smf.ols('mpg ~ acceleration', data=dc).fit()
sm.graphics.influence_plot(results)
plt.title("acceleration")
#plt.show()

results = smf.ols('mpg ~ year', data=dc).fit()
sm.graphics.influence_plot(results)
plt.title("year")
#plt.show()

results = smf.ols('mpg ~ origin', data=dc).fit()
sm.graphics.influence_plot(results)
plt.title("origin")
#plt.show()

model = smf.ols(formula='mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin', data=dc).fit()
summary = model.summary()
print(summary)


