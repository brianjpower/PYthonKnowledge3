import numpy as np
import pandas as pd
from pandas import DataFrame, Series

#Read in prostate and heart data

prostate = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/prostate.data',sep='\t')
SA = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data')


print(prostate.head(10))


print(prostate.describe())

#First drop the non-numeric columns and standardise the data
prostate_numeric = prostate.drop(['Unnamed: 0','train'],axis=1)
prostate_std = (prostate_numeric-prostate_numeric.mean())/prostate_numeric.std()

print(prostate_std)

#Now can identify outliers
print(prostate_std[(np.abs(prostate_std)>3).any(1)])



#%% md
#### Exercise 1

#Standardise the South African heart disease dataset and identify any
# outliers that are at least 4 standard deviations away from the mean.
# How many of these outliers are there in the tobacco column?
#
print(SA.head(10))
print(SA.describe())
SA_numeric = SA.drop(['row.names','famhist'],axis=1)


SA_std = (SA_numeric - SA_numeric.mean())/SA_numeric.std()
print(SA_std[(np.abs(SA_std)>4).any(1)])
print(SA_std[(np.abs(SA_std['tobacco'])>4)].count()[0])


import scipy.stats as stats

#two sample t-test
#%% md
## Two-sample t-test

#Let's see whether lpsa (log prostate specific antigen - the key
#response variable) is significantly different between those aged
#65 and under versus those aged over 65.

#We first need to split the data into two groups.
#Check if

lpsa_gt65 = prostate[prostate['age']>65]['lpsa']
lpsa_lt65 = prostate[prostate['age']<=65]['lpsa']
#print(lpsa_gt65.count()[0])
#print(lpsa_lt65.count()[0])
print(stats.ttest_ind(lpsa_gt65,lpsa_lt65))

#exercise 2

lsps_train = prostate.lpsa[prostate.train=="T"]
lpsa_test = prostate.lpsa[prostate.train=="F"]
print(stats.ttest_ind(lsps_train,lpsa_test,equal_var=False))

#If don't assume that the sample is drawn from a normal dist
#use mann whitney

print(stats.mannwhitneyu(lpsa_gt65, lpsa_lt65))

#%% md
## Kolmogorov-Smirnov

#Suppose we want to check the shape of the distribution of lpsa.
# We should probably start by creating a histogram
#
import matplotlib.pyplot as plt
plt.figure()
prostate_std.lpsa.hist()
plt.show()

#%% md
#We can run a Kolmogorov-Smirnov test to test the
# distribution shape, using the scipt.stats command `kstest`.
#
print(stats.kstest(prostate.lpsa, 'norm'))

#Looks alarming, check the standardised dist
print(stats.kstest(prostate_std.lpsa, 'norm'))

print(stats.pearsonr(prostate.lpsa, prostate.lweight))

#(0.43331938249262, 9.27649916588068e-06)
#1.
#    - The value `0.433` indicates a **moderate positive linear relationship**
#    between the two variables.


#1.
#    - The p-value is approximately **0.0000093**, which is **extremely small**
#    and statistically significant.



##QQ plots

plt.figure()
stats.probplot(prostate.lpsa, dist='norm',plot=plt);
plt.show()

#Linear regression with statsmodels api

import statsmodels.api as sm
mod = sm.OLS(prostate_std.lpsa,prostate_std.drop('lpsa',axis=1))
res = mod.fit()
print(res.summary())

X_with_const = prostate_std.drop('lpsa',axis=1)
X_with_const.insert(0,'intercept',1)

mod = sm.OLS(prostate_std.lpsa,X_with_const)
res = mod.fit()
print(res.summary())

X_red = prostate_std[['lcavol','lweight']]
X_red.insert(0,'intercept',1)

mod = sm.OLS(prostate_std.lpsa,X_red)
res = mod.fit()
print((res.summary()))

#more R like way of performing fit

import statsmodels.formula.api as smf
mod2 = smf.ols(formula='lpsa ~ lcavol + lweight', data=prostate_std)
res2 = mod2.fit()
print(res2.summary())


#%% md
#This method includes an intercept by default. You should read through the help files for `ols` and `fit` to see all of the available input arguments. Additional arguments include `method` which allows you change the fitting method and `subset` which assigns the subset of data to be used in the model it.

# numbers. In addition, this method 2 allows you to easily include interaction terms and categorical terms.

mod6 = smf.ols(formula='lpsa ~ lcavol + pow(lcavol,2)', data=prostate_std).fit()
print(mod6.summary())
mod_summary = DataFrame({'preds':mod6.predict(),'resids':mod6.resid})
mod_summary.plot('preds','resids',kind='scatter')
plt.show()

#scatter plot show the variance of the residuals is the same for all values of X, key assumption
#of linear regressions

##Logistic regression
#Similar to linear regression except the response endogeneous variables is binary and
#treated as a Bernoulli distributed variable

SA_numeric = SA.drop(['row.names','famhist','chd'],axis=1)
SA_std = (SA_numeric-SA_numeric.mean())/SA_numeric.std()
SA_std.insert(0,'intercept',1)

mod = sm.Logit(SA.chd,SA_std)
logit1 = mod.fit()
print(logit1.summary())


logit2 = sm.Logit(SA.chd, SA_std['age']).fit()
new_age = DataFrame({'age': np.linspace(-3,3,100)})
new_preds = logit2.predict(new_age)
new_age['preds'] = new_preds

plt.figure()
new_age.plot('age','preds')
plt.plot(SA_std['age'],SA.chd,'r+')
plt.ylim(-0.05,1.05)
plt.show()

logit_smf = smf.logit(formula='chd ~ sbp + tobacco + famhist', data=SA).fit()
print(logit_smf.summary())

#%% md
#### Exercise 5
#Fit each of the models below to the SA heart disease data. Which relationship results in the best fit (has the highest Pseudo-R$^2$ value)?
#* 'chd ~ adiposity + tobacco + ldl'
#* 'chd ~ sbp + tobacco + famhist'
#* 'chd ~ adiposity + tobacco'
#* 'chd ~ tobacco + ldl'

#logit_smf1 = smf.logit(formula='chd ~  adiposity + tobacco + ldl', data = SA).fit()
#print(logit_smf1.summary())
logit_smf2 = smf.logit(formula='chd ~ sbp + tobacco + famhist', data = SA).fit()
print(logit_smf2.summary())
#logit_smf3 = smf.logit(formula='chd ~ adiposity + tobacco', data = SA).fit()
#print(logit_smf3.summary())
#logit_smf4 = smf.logit(formula='chd ~ tobacco + ldl', data = SA).fit()
#print(logit_smf4.summary())





