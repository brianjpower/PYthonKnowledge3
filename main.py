import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sqlalchemy.dialects.mssql.information_schema import columns
from wbdata.client import DataFrame

"""
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

#########################Section 10 - Machine Learning########################################
print("==========================Section 10==============================")

from sklearn import datasets
diabetes = datasets.load_diabetes()
#print(diabetes.feature_names)
#print(diabetes.target)

X_raw = DataFrame(diabetes.data,columns=diabetes.feature_names)
X_std = (X_raw - X_raw.mean())/X_raw.std()
y = Series(diabetes.target)

#Append some noise terms
np.random.seed(123)
X = X_std.join(DataFrame(np.random.randn(len(y), 100)))

print(X.describe())

#Exercise 1 - load and tidy Boston dataset

boston = datasets.load_boston()
print(boston.feature_names)
X_raw = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = Series(boston.target)
X_std = (X_raw-X_raw.mean())/X_raw.std()

print(X_std.head())
print(Y.head())

train_size = int(len(X_raw)*0.75)
print(train_size)
np.random.seed(123)
train_select = np.random.permutation(range(len(Y)))
X_train = X_raw.iloc[train_select[:train_size],:].reset_index(drop=True)
X_test = X_raw.iloc[train_select[train_size:],:].reset_index(drop=True)
Y_train = Y.iloc[train_select[:train_size]].reset_index(drop=True)
Y_test = Y.iloc[train_select[train_size:]].reset_index(drop=True)

#%% md
#### Exercise 2
#Split the Boston house price data into training and test sets,
# of size 380 and 126, respectively. Use 99 as the random seed value
# for your permutation. Is the mean house price roughly the same in both
# training and test sets?
#
boston = datasets.load_boston()
#print(boston.feature_names)
Xb_raw = pd.DataFrame(boston.data, columns=boston.feature_names)
Yb = Series(boston.target)
Xb_std = (Xb_raw-Xb_raw.mean())/Xb_raw.std()
bost_train_size = int(len(Yb)*(0.75))
print(f"Boston train size is {bost_train_size}")
np.random.seed(99)
train_select = np.random.permutation(range(len(Yb)))
Xb_train = Xb_raw.iloc[train_select[:bost_train_size],:].reset_index(drop=True)
Xb_test = Xb_raw.iloc[train_select[bost_train_size:],:].reset_index(drop=True)
Yb_train= Yb.iloc[train_select[:bost_train_size]].reset_index(drop=True)
Yb_test = Yb.iloc[train_select[bost_train_size:]].reset_index(drop=True)

print(f"Average house price train is {Yb_train.mean()}\n")
print(f"Average house price test is {Yb_test.mean()}\n")
comp_house_price = stats.ttest_ind(Yb_train, Yb_test)
print(comp_house_price)
print(Xb_train.columns)

#linear regression


from sklearn import linear_model

lin_reg = linear_model.LinearRegression()
#Use the training data to fit a model
lin_reg.fit(X_train, Y_train)
model = lin_reg.fit(X_train, Y_train)
print(lin_reg.coef_)
#apply the model to predict the Y values from the test data
reg_test_pred = lin_reg.predict(X_test)
#Asses the performance of the model by plotting predicted versus actual values
fig = plt.figure()
plt.plot(reg_test_pred,Y_test,'o')
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()

#The MSE can help us check the model performance
MSE_reg = np.mean(pow((reg_test_pred - Y_test), 2))
print(max(Y_test) - min(Y_test))
print(MSE_reg)

corr_reg = np.corrcoef(reg_test_pred, Y_test)[0,1]
print(corr_reg)

print(f"R^2 for linear regression is {lin_reg.score(X_test, Y_test)}")

#Boston house prices prediction
Yb_pred = lin_reg.predict(Xb_test)
print(len(Yb_pred))
print(len(Yb_test))

fig = plt.figure()


plt.plot(Yb_pred,Yb_test,'o')
plt.xlabel("Predicted House Prices")
plt.ylabel("Actual House Prices")
plt.show()

MSE_reg = np.mean(pow((Yb_pred - Yb_test), 2))
print(f"The range is {max(Yb_test) - min(Yb_test)}")
print(MSE_reg)

corr_reg = np.corrcoef(Yb_pred, Yb_test)[0,1]
print(corr_reg)
print(Yb_pred)
print(pd.Series(Yb_test))
print(f"R^2 for linear regression is {lin_reg.score(Xb_test,Yb_test)}")

lasso = linear_model.LassoCV(cv=2)
lasso.fit(X_train, Y_train)
print(lasso.coef_)


lasso = linear_model.LassoCV(cv=10)
lasso.fit(X_train, Y_train)
print(lasso.coef_)

lasso_test_pred = lasso.predict(X_test)
fig = plt.figure()
plt.plot(Y_test,lasso_test_pred,'kx')
plt.plot(plt.xlim(), plt.ylim(), ls="--")
MSE_lasso = np.mean(pow((lasso_test_pred - Y_test),2))
print(MSE_lasso)
plt.show()


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, Y_train)
rf_test_pred = rf.predict(X_test)
fig = plt.figure()
plt.plot(Y_test,rf_test_pred,'kx')
plt.plot(plt.xlim(), plt.ylim(), ls="--")
plt.show()
MSE_rf = np.mean(pow((rf_test_pred- Y_test),2))
print(MSE_rf)

"""
#Classification

#Read in the South Africa dataset again and split into training and test datasets
SA = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data')

X_raw = SA.drop(['row.names','famhist','chd'],axis=1)
X_raw_std = (X_raw-X_raw.mean())/X_raw.std()
y = SA.chd
np.random.seed(123)
X = X_raw_std.join(DataFrame(np.random.randn(len(y), 100)))

train_size = int(len(X_raw)*0.75)
np.random.seed(123)
train_select = np.random.permutation(range(len(y)))
X_train = X.iloc[train_select[:train_size],:].reset_index(drop=True)
X_test = X.iloc[train_select[train_size:],:].reset_index(drop=True)
y_train = y[train_select[:train_size]].reset_index(drop=True)
y_test = y[train_select[train_size:]].reset_index(drop=True)

#logistic regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
logreg_test_pred = logreg.predict(X_test)
logreg_cross = pd.crosstab(logreg_test_pred,y_test)
#print confusion matrix
print(logreg_cross)
logreg_misclas = (logreg_cross.iloc[0,1]+logreg_cross.iloc[1,0])/np.sum(logreg_cross.values)*100
print(f"Misclassification rate is: {logreg_misclas}%")
#%% md
#`predict` predicts the category (0 or 1 in this case), while `predict_proba`
# will give the probability of the result being in each category.

#
print(logreg_test_pred[:10],'\n')
print("==================================\n")
logreg_test_prob = logreg.predict_proba(X_test)
print("==================================\n")
print(logreg_test_prob[:10])
print("==================================\n")
print(logreg_test_pred[:10])

#Lasso logistic regression
lassologreg = LogisticRegression(solver='saga',penalty='l1',C=0.1)
lassologreg.fit(X_train, y_train)
lassologreg_test_pred = lassologreg.predict(X_test)
lassologreg_cross = pd.crosstab(lassologreg_test_pred,y_test)
print(lassologreg_cross)

lassologreg_misclas = (lassologreg_cross.iloc[0,1]+lassologreg_cross.iloc[1,0])/np.sum(lassologreg_cross.values)*100
print(logreg_misclas)

#random forest classification
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf_fit = rf_clf.fit(X_train, y_train)
rf_clf_test_pred = rf_clf_fit.predict(X_test)
rf_clf_cross = pd.crosstab(rf_clf_test_pred,y_test)
print(rf_clf_cross)

rf_clf_misclas = (rf_clf_cross.iloc[0,1]+rf_clf_cross.iloc[1,0])/np.sum(rf_clf_cross.values)*100
print(rf_clf_misclas)
"""
from sklearn import datasets
iris = datasets.load_iris()

x = DataFrame(iris.data,columns=iris.feature_names)
x_std = (x - x.mean())/x.std()
print(x_std.describe())
y = Series(iris.target)
#print(y)
train_size = int(len(iris)*0.75)
np.random.seed(123)
train_select = np.random.permutation(range(len(y)))

#X_train = X.iloc[train_select[:train_size],:].reset_index(drop=True)
#X_test = X.iloc[train_select[train_size:],:].reset_index(drop=True)
#y_train = y[train_select[:train_size]].reset_index(drop=True)
#y_test = y[train_select[train_size:]].reset_index(drop=True)

xtrain = x_std.iloc[train_select[:train_size],:].reset_index(drop=True)
xtest = x_std.iloc[train_select[train_size:],:].reset_index(drop=True)
ytrain = y.iloc[train_select[:train_size]].reset_index(drop=True)
ytest = y.iloc[train_select[train_size:]].reset_index(drop=True)

from sklearn.ensemble import RandomForestClassifier


rf_iris = RandomForestClassifier(n_estimators=20)
rf_iris_fit = rf_iris.fit(xtrain, ytrain)
rf_iris_test_pred = rf_iris.predict(xtest)
conf = pd.crosstab(rf_iris_test_pred,ytest)
print(conf)

rf_iris_misclas = (conf.iloc[0,1]+conf.iloc[0,2]+conf.iloc[1,0] + conf.iloc[2,0] + conf.iloc[1,2] + conf.iloc[2,1])/np.sum(conf.values)*100
print(f"The iris misclassification rate is {rf_iris_misclas}%")

from sklearn.metrics import roc_curve, auc
roc_lr = roc_curve(y_test, logreg_test_prob)
lr_auc = auc(roc_lr[0],roc_lr[1])
roc_lassolr = roc_curve(y_test, lassologreg_test_prob)
lassolr_auc = auc(roc_lassolr[0],roc_lassolr[1])
roc_rf = roc_curve(y_test, rf_test_prob)
rf_auc = auc(roc_rf[0],roc_rf[1])
roc_svm = roc_curve(y_test, svm_test_prob)
svm_auc = auc(roc_svm[0],roc_svm[1])
"""

#Additional exercises
from sklearn import datasets
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

digits = datasets.load_digits()

choose_row = 100
plt.figure()
plt.gray()
plt.matshow(digits.images[choose_row])
plt.title(digits.target[choose_row])
plt.show()

X_raw = DataFrame(digits.data)
X = (X_raw - X_raw.mean())/X_raw.std().replace(0,1)
y = Series(digits.target)

np.random.seed(99)
n_train = int(len(y)/2)
n_val =  int(3*len(y)/4)
inds = np.random.permutation(range(len(y)))
X_train = X.iloc[inds[:n_train],:].reset_index(drop=True)
X_val = X.iloc[inds[n_train:n_val],:].reset_index(drop=True)
X_test = X.iloc[inds[n_val:],:].reset_index(drop=True)
y_train = y.iloc[inds[:n_train]].reset_index(drop=True)
y_val = y.iloc[inds[n_train:n_val]].reset_index(drop=True)
y_test = y.iloc[inds[n_val:]].reset_index(drop=True)

