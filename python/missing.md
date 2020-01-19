#  Types of Missingness

* Missing completely at random (MCAR): 

When data are MCAR, missing cases are, on average, identical to non-missing cases, with respect to the model. Ignoring the missingness will reduce the power of the analysis, but will not bias inference.

* Missing at random (MAR): 

Missing data depends (usually probabilistically) on measured values, and hence can be modeled by variables observed in the data set. Accounting for the values which “cause” the missing data will produce unbiased results in an analysis.

* Missing not at random(MNAR): 

Missing data depend on unmeasured or unknown variables. There is no information available to account for the missingness.

The very best-case scenario for using complete case analysis, which corresponds to MCAR missingness, results in a loss of power due to the reduction in sample size. The analysis will lose the information contained in the non-missing elements of a partially-missing record. When data are not missing completely at random, inferences from complete case analysis may be biased due to systematic differences between missing and non-missing records that affects the estimates of key parameters.

One alternative to complete case analysis is to simply fill (impute) the missing values with a reasonable guess a the true value, such as the mean, median or modal value of the fully-observed records. This imputation, while not recovering any information regarding the missing value itself for use in the analysis, does provide a mechanism for including the non-missing values in the analysis, thereby making use of all available information.


*  How would you deal with the missing values in "country"?

You have to predict conversion rate on Airbnb using user country as one of the input variables.

Country is missing in the dataset if the user chooses to not select her country when she creates her profile after signing up

There are many ways to deal with missing values if you go by the books. The most cited ones are to replace
them with the median/average or build a model to predict them and then use the predicted values.

In practice, these things tend to make more sense in a book than in real life. Firstly, because the high high
majority of missing values in tech companies come from the fact that the user has chosen to not give that kind
of information: by not filling out her profile, or by changing her privacy settings, or even by deleting her cookies.
And that's crucial information for your predictive model that you would lose, if you were replacing the missing
values.

Secondly, if you could predict the missing values using other variables, then just include also those other variables and use a model that works well with correlated variables, like tree-based models for instance. Using
the other variables your model would still be able to extract country-related informations + it could extract
information from the fact that the user chose to not provide that information.
The safest approach is therefore to use some label like "no_country" when the value is missing. And if by using
other variables you can predict the actual country, for instance using a combination of ip address and service
provider, then include those variables too in your training set. So for each user you can know both where she is
based and whether she chose to tell you her country.

This approach (use mean to fill na) may be reasonable under the MCAR assumption, but may induce bias under a MAR scenario, whereby missing values may differ systematically relative to non-missing values, making the particular summary statistic used for imputation biased as a mean/median/modal value for the missing values.

Beyond this, the use of a single imputed value to stand in place of the actual missing value glosses over the uncertainty associated with this guess at the true value. Any subsequent analysis procedure (e.g. regression analysis) will behave as if the imputed value were observed, despite the fact that we are actually unsure of the actual value for the missing variable. The practical consequence of this is that the variance of any estimates resulting from the imputed dataset will be artificially reduced.

```python
from sklearn.impute import SimpleImputer

# mean

imp = SimpleImputer(strategy='mean')

imp.fit(test_scores)

imp.transform(test_scores)[:3]

# another way
s_imputed = t.sib.fillna(t.sib.mean())



# mode 

mode_imp = SimpleImputer(strategy = 'most_frequent')

mode_imp.fit(test_scores)

mode_imp.transform(test_scores)[:3]

```



# Multiple Imputation

**single imputation didn't take uncertainty of imputation into account (treat as observed)**

One robust alternative to addressing missing data is multiple imputation (Schaffer 1999, van Buuren 2012). It produces unbiased parameter estimates, while simultaneously accounting for the uncertainty associated with imputing missing values. It is conceptually and mechanistically straightforward, and produces complete datasets that may be analyzed using any statistical methodology or software one chooses, as if the data had no missing values to begin with.

Multiple imputation generates imputed values based on a regression model. This regression model will help us generate reasonable values, particularly if data are MAR, since it uses information in the dataset that may be informative in predicting what the true value may be. 

Ideally, we want predictor variables that are correlated with the missing variable, and with the mechanism of missingness, if any. For example, one might be able to use test scores from one subject to predict missing test scores from another; or, the probability of income reporting to be missing may vary systematically according to the education level of the individual.

To see if there is **any potential information among the variables in our dataset to use for imputation**, it is helpful to calculate the **pairwise correlation** between all the variables. Since we have discrete variables in our data, the Spearman rank correlation coefficient is appropriate.

```python
# check what vars correlated with our var of interest (categorical)

# 这里很有启发：
test.dropna().corr(method='spearman').round(2)

# we want to impute a binary outcome

from sklearn.linear_model import LogisticRegression

# only include complete var to infer 'mother_hs'
impute_subset = test.drop(labels = ['f','p','s'].axis=1)

y = impute_subset.pop('mother_hs').values

# scale vars
x = preprocessing.StandardScalar().fit_transform(impute_subset.astype(float)）

missing = np.isnan(y)

mod = LogisticRegression()

mod.fit(X[~missing],y[~missing])

mother_hs_pred = mod.predict(x[missing])

# try several models (diff penalization) and take avg to hopefully provide more bobus

mod2 = LogisticRegression(C=1,penalty='l1')

mod2.fit(x[~missing],y[~missing])
mod2.predict(x[missing])

# try 3-10 estimates

mother_hs_imp = []

for C in 0.1,0.4,2:
	mod = LogisticRegression(C=C,penalty='l1')
	mod.fit(x[~missing],y[~missing])
	imputed = mod.predict(x[missing])
	mother_hs_imp.append(imputed)
	
```

# plot svm stuff

```python
from sklearn import svm

grape = wine.pop('region')

y = grape.values

wine.columns = attributes

x = wine[['Alcohol','Proline']].values

svc = svm.SVC(kernel='linear')

svc.fit(x,y)

# plot regularized

def plot_regularized(power,ax):
	svc = svm.SVC(kernel='linear',C=10**power)
	plot_estimator(svc,X,y,ax=ax)
	ax.scatter(svc.support_vectors_[:,0],svc.support_vectors_[:,1],s=80,
			facecolors ='none',edgecolors='w',linewidths=2,zorder=10)
	ax.set_title('Power={}'.format(power))

fig, axes = plt.subplots(2,3,figsize=(12,10))
for power,ax in zip(range(-2,4),axes.ravel()):
	plot_regularized(power,ax)
	
	
# kernels: linear, poly, rbf, sigmoid, precomputed

def plot_poly_svc(degree=3,ax=None):
	svc_poly = svm.SVC(kernel='poly',degree= degree)
	plot_estimator(svc_poly,x,y,ax=ax)
	ax.scatter(svc_poly.support_vectors_[:,0],svc_poly.support_vectors_[:1],
		s = 80, facecolors = 'none',linewidths = 2, zorder = 10)
	ax.set_title('poly {}'.format(degree))
	
fig,axes = plt.subplots(2,3,figsize=(12,10))
for deg,ax in zip(range(1,7),axes.ravel()):
	plot_poly_svc(deg,ax)
	
```


# cv

```python
 quality of fit in cross-validaiton

svc_lin = svm.SVC(kernel='linear',C=2)
svc_lin.fit(x,y)
svc_lin.score(x,y)


svc_poly = svm.SVC(kernel='poly',degree = 3)
svc_poly.fit(x,y)
svc_poly.score(x,y)

svc_rbf = svm.SVC(kernel='rbf',gamma=1e-2)
svc_rbf.fit(x,y)
svc_rbf.score(x,y)
```


# eval model

Each estimator in scikit-learn has a default estimator score method, which is an evaluation criterion for the problem they are designed to solve. For the SVC class, this is the mean accuracy, as shown above.

Alternately, if we use cross-validation, you can specify one of a set of built-in scoring metrics. For classifiers such as support vector machines, these include:

Each estimator in scikit-learn has a default estimator score method, which is an evaluation criterion for the problem they are designed to solve. For the SVC class, this is the mean accuracy, as shown above.

accuracy : sklearn.metrics.accuracy_score

average_precision : sklearn.metrics.average_precision_score

f1 : sklearn.metrics.f1_score

precision : sklearn.metrics.precision_score

recall : sklearn.metrics.recall_score

roc_auc : sklearn.metrics.roc_auc_score

Regression models can use appropriate metrics, like mean_squared_error or r2.

Finally, one can specify arbitrary loss functions to be used for assessment. The metrics module implements functions assessing prediction errors for specific purposes.


		
# can make your own score

```python
def custom_loss(observed,predicted):
	diff = np.abs(observed - predicted).max()
	return np.log(diff+1)
	

from sklearn.metrics import make_scorer

custom_scorer = make_scorer(custom_loss,greater_is_better = False)

# implement cv

from sklearn import model_selection

xtrain,xtest,ytrain,ytest = model_selection.train_test_split(wine.values,grape.values,
		test_size = 0.4, random_state=0)
		
x_train.shape

f = svm.SVC(kernel='linear',C=1)
f.fit(x_train,y_train)
f.score(x_test,y_test)

# est accuracy of a linear svm by split data, fit model and compute score for diff splits

scores = model_selection.cross_validation_score(f,wine.values,grape.values,cv=5)

scores

print("acc: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

model_selection.cross_val_score(f,wine.values,grape.values,cv=5,scoring = 'f1_weighted')

from sklearn.metrics import confusion_matrix

svc_poly = svm.SVC(kernel='poly',degree=3).fit(x_train,y_train)

confusion_matrix(y_test,svc_poly.predict(x_test))


scores = model_selection.cross_val_score(f,wine.values,grape.values,cv=5)
```





