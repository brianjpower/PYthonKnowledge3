#Additional Exercises

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import numpy.random as npr

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

prostate = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/prostate.data',sep='\t')
print(prostate.head())

prostate2 = prostate[['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']]
print(prostate2.head())
prostate2_std = (prostate2 - prostate2.mean()) / prostate2.std()
print(prostate2_std.describe())
print(prostate2_std.head())

y = Series(prostate['lpsa'])
y_std = (y -y.mean()) / y.std()
print(y_std.describe())

mod = smf.ols(formula="y_std~lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45", data=prostate2_std).fit()
print(mod.summary())


prostate_aic = smf.ols(formula="y_std ~ 1", data=prostate).fit()
print(prostate_aic.aic)

print(prostate2_std.iloc[:,0])

print(prostate2_std.shape[1])
print(prostate_aic.summary())

for i in range(prostate2_std.shape[1]):
    print(prostate2_std.iloc[:,i])



def forwardAIC(dataframe,y):
    aic_values = []
    aic_keep = []
    aic_index = []
    aic_base = smf.ols(formula="y ~ 1", data=dataframe).fit().aic
    print(aic_base)
    for i in range(dataframe.shape[1]):
        x = dataframe.iloc[:,i]
        mod = smf.ols(formula="y ~ x", data=dataframe).fit()
        aic_values.append(mod.aic)
        #aic_keep.append(mod.aic)
        if(i==0):
            if(mod.aic < aic_base):
                aic_keep.append(mod.aic)
                aic_index.append(i)
        else:
            if(mod.aic < aic_values[i-1]):
                aic_keep.append(mod.aic)
                aic_index.append(i)
    return aic_index,aic_values,aic_keep

print(forwardAIC(prostate2_std,y_std))


def forwardAIC2(X, y):
    # X here must be a matrix with the first column just a constant.
    # The remaining columns should be the explanatory variables
    # y should be a series containing the response variable

    # First, fit a model with just a constant:
    mod = sm.OLS(y, X.iloc[:, 0]).fit()
    best_aic = mod.aic

    # Create a while loop to run through the model
    bad_model = True
    # Get the column indices of the chosen vars so far
    chosen_vars = [0]
    # Get the column indices of the remaining vars so far
    remaining_vars = range(1, X.shape[1])
    while (bad_model):
        # Loop through all the remaining vars
        curr_aic = np.empty(len(remaining_vars))
        curr_aic_diff = np.empty(len(remaining_vars))
        for count, i in enumerate(remaining_vars):
            curr_vars = np.append(chosen_vars, i)
            curr_mod = sm.OLS(y, X.iloc[:, curr_vars]).fit()
            curr_aic[count] = curr_mod.aic
            curr_aic_diff[count] = curr_mod.aic - best_aic
        # If the models are better at least one of these should be negative
        if len(remaining_vars) == 0:
            bad_model = False
        elif np.min(curr_aic_diff) > 0:
            bad_model = False
        else:
            best_var = remaining_vars[np.argmin(curr_aic_diff)]
            best_aic = curr_aic[np.argmin(curr_aic_diff)]
            chosen_vars = np.append(chosen_vars, best_var)
            remaining_vars = [x for x in remaining_vars if x != best_var]
    return chosen_vars


ans = forwardAIC2(prostate2_std,y_std)
print(ans)
print(prostate2_std.columns)


