
# Lesson 1

CRISP-DM

Therefore, all steps of CRISP-DM were not necessary for these first two questions. CRISP-DM states 6 steps:

1. Business Understanding

2. Data Understanding

3. Prepare Data

4. Data Modeling

5. Evaluate the Results

6. Deploy

CRISP-DM

In working with missing values, categorical variables, and building out your model, it was probably easy to lose site of the big picture of the process. Let's take a quick second to recap that here, and pull together the results you should have arrived through your analysis.

1. Business Understanding

How do I break into the field?
What are the placement and salaries of those who attended a coding bootcamp?
How well can we predict an individual's salary? What aspects correlate well to salary?
How well can we predict an individual's job satisfaction? What aspects correlate well to job satisfaction?
2. Data Understanding

Here we used the StackOverflow data to attempt to answer our questions of interest. We did 1. and 2. in tandem in this case, using the data to help us arrive at our questions of interest. This is one of two methods that is common in practice. The second method that is common is to have certain questions you are interested in answering, and then having to collect data related to those questions.

3. Prepare Data

This is commonly denoted as 80% of the process. You saw this especially when attempting to build a model to predict salary, and there was still much more you could have done. From working with missing data to finding a way to work with categorical variables, and we didn't even look for outliers or attempt to find points we were especially poor at predicting. There was ton more we could have done to wrangle the data, but you have to start somewhere, and then you can always iterate.

4. Model Data

We were finally able to model the data, but we had some back and forth with step 3. before we were able to build a model that had okay performance. There still may be changes that could be done to improve the model we have in place. From additional feature engineering to choosing a more advanced modeling technique, we did little to test that other approaches were better within this lesson.

5. Results

Results are the findings from our wrangling and modeling. They are the answers you found to each of the questions.

6. Deploy

Deploying can occur by moving your approach into production or by using your results to persuade others within a company to act on the results. Communication is such an important part of the role of a data scientist.





The first two steps of CRISP-DM are:

1. Business Understanding - this means understanding the problem and questions you are interested in tackling in the context of whatever domain you're working in. Examples include

How do we acquire new customers?
Does a new treatment perform better than an existing treatment?
How can improve communication?
How can we improve travel?
How can we better retain information?

2. Data Understanding - at this step, you need to move the questions from Business Understanding to data. You might already have data that could be used to answer the questions, or you might have to collect data to get at your questions of interest

```python
num_rows = df.shape[0] #Provide the number of rows in the dataset
num_cols = df.shape[1] #Provide the number of columns in the dataset

no_nulls = set(df.columns[df.isnull().mean()==0])#Provide a set of columns with 0 missing values.

most_missing_cols = set(df.columns[df.isnull().mean() > 0.75])#Provide a set of columns with more than 75% of the values missing
status_vals = df.Professional.value_counts()#Provide a pandas series of the counts for each Professional status

# The below should be a bar chart of the proportion of individuals in each professional category if your status_vals
# is set up correctly.

(status_vals/df.shape[0]).plot(kind="bar");
plt.title("What kind of developer are you?");
```

# Business and Data Understanding

Business Questions

How do I break into the field? (Education)

What are the placement and salaries of those who attended a coding bootcamp?

How well can we predict an individual's salary? What aspects correlate well to salary?

How well can we predict an individual's job satisfaction? What aspects correlate well to job satisfaction?

Data Understanding

You will be using the Stackoverflow survey data to get some insight into each of these questions. In the rest of the lesson, you can work along with me to answer these questions, or you can take your own approach to see if the conclusions you draw are similar to those you would find throughout this lesson.

```python
def get_description(column_name, schema=schema):
    '''
    INPUT - schema - pandas dataframe with the schema of the developers survey
            column_name - string - the name of the column you would like to know about
    OUTPUT - 
            desc - string - the description of the column
    '''
    desc = list(schema.loc[schema['Column'] == column_name,'Question'])[0]
    return desc
#Check your function against solution - you shouldn't need to change any of the below code
descrips = set(get_description(col) for col in df.columns)
t.check_description(descrips)
```



```python

def total_count(df, col1, col2, look_for):
    '''
    INPUT:
    df - the pandas dataframe you want to search
    col1 - the column name you want to look through
    col2 - the column you want to count values from
    look_for - a list of strings you want to search for in each row of df[col]

    OUTPUT:
    new_df - a dataframe of each look_for with the count of how often it shows up
    '''
    new_df = defaultdict(int)
    #loop through list of ed types
    for val in look_for:
        #loop through rows
        for idx in range(df.shape[0]):
            #if the ed type is in the row add 1
            if val in df[col1][idx]:
                new_df[val] += int(df[col2][idx])
    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns = [col1, col2]
    new_df.sort_values('count', ascending=False, inplace=True)
    return new_df

possible_vals = ["Take online courses", "Buy books and work through the exercises", 
                 "None of these", "Part-time/evening courses", "Return to college",
                 "Contribute to open source", "Conferences/meet-ups", "Bootcamp",
                 "Get a job as a QA tester", "Participate in online coding competitions",
                 "Master's degree", "Participate in hackathons", "Other"]

def clean_and_plot(df, title='Method of Educating Suggested', plot=True):
    '''
    INPUT 
        df - a dataframe holding the CousinEducation column
        title - string the title of your plot
        axis - axis object
        plot - bool providing whether or not you want a plot back
        
    OUTPUT
        study_df - a dataframe with the count of how many individuals
        Displays a plot of pretty things related to the CousinEducation column.
    '''
    study = df['CousinEducation'].value_counts().reset_index()
    study.rename(columns={'index': 'method', 'CousinEducation': 'count'}, inplace=True)
    study_df = t.total_count(study, 'method', 'count', possible_vals)

    study_df.set_index('method', inplace=True)
    if plot:
        (study_df/study_df.sum()).plot(kind='bar', legend=None);
        plt.title(title);
        plt.show()
    props_study_df = study_df/study_df.sum()
    return props_study_df
    
props_df = clean_and_plot(df)
```


```python
def higher_ed(formal_ed_str):
    '''
    INPUT
        formal_ed_str - a string of one of the values from the Formal Education column
    
    OUTPUT
        return 1 if the string is  in ("Master's degree", "Doctoral", "Professional degree")
        return 0 otherwise
    
    '''
    if formal_ed_str in ("Master's degree", "Doctoral", "Professional degree"):
        return 1
    else:
        return 0
    

df["FormalEducation"].apply(higher_ed)[:5] #Test your function to assure it provides 1 and 0 values for the df
```

```python
#Check your subset is correct - you should get a plot that was created using pandas styling
#which you can learn more about here: https://pandas.pydata.org/pandas-docs/stable/style.html
ed_1 = df[df['HigherEd']==1]# Subset df to only those with HigherEd of 1
ed_0 = df[df['HigherEd']==0]# Subset df to only those with HigherEd of 0
ed_1_perc = clean_and_plot(ed_1, 'Higher Formal Education', plot=False)
ed_0_perc = clean_and_plot(ed_0, 'Max of Bachelors Higher Ed', plot=False)

comp_df = pd.merge(ed_1_perc, ed_0_perc, left_index=True, right_index=True)
comp_df.columns = ['ed_1_perc', 'ed_0_perc']
comp_df['Diff_HigherEd_Vals'] = comp_df['ed_1_perc'] - comp_df['ed_0_perc']

# this is interesting
comp_df.style.bar(subset=['Diff_HigherEd_Vals'], align='mid', color=['#d65f5f', '#5fba7d'])
```

```python
df.groupby('CompanySize')['JobSatisfaction'].mean().sort_values()
```


All Data Science Problems Involve
1. Curiosity.

2. The right data.

3. A tool of some kind (Python, Tableau, Excel, R, etc.) used to find a solution (You could use your head, but that would be inefficient with the massive amounts of data being generated in the world today).

4. Well communicated or deployed solution.

Extra Useful Tools to Know But That Are NOT Necessary for ALL Projects

Deep Learning

Fancy machine learning algorithms

With that, you will be getting a more in depth look at these items, but it is worth mentioning (given the massive amount of hype) that they do not solve all the problems. Deep learning cannot turn bad data into good conclusions. Or bad questions into amazing results.


The variables we use to predict are commonly called X (or an X matrix). The column we are interested in predicting is commonly called y (or the response vector).

In this case X is all the variables in the dataset that are not salary, while y is the salary column in the dataset.

On the next page, you will see what happens when we try to use sklearn to fit a model to the data, and we will do some work to get useful predictions out of our sklearn model.

```python
# just plot cor of numerical

sns.heatmap(df.corr(), annot=True, fmt=".2f");

```


There are two main 'pain' points for passing data to machine learning models in sklearn:

Missing Values

Categorical Values

Sklearn does not know how you want to treat missing values or categorical variables, and there are lots of methods for working with each. For this lesson, we will look at common, quick fixes. These methods help you get your models into production quickly, but thoughtful treatment of missing values and categorical variables should be done to remove bias and improve predictions over time.

Three strategies for working with missing values include:

1. We can remove (or “drop”) the rows or columns holding the missing values.

Though dropping rows and/or columns holding missing values is quite easy to do using numpy and pandas, it is often not appropriate.

Understanding why the data is missing is important before dropping these rows and columns. In this video you saw a number of situations in which dropping values was not a good idea. These included

Dropping data values associated with the effort or time an individual put into a survey.

Dropping data values associated with sensitive information.

In either of these cases, the missing values hold information. A quick removal of the rows or columns associated with these missing values would remove missing data that could be used to better inform models.

Instead of removing these values, we might keep track of the missing values using indicator values, or counts associated with how many questions an individual skipped.

In the last video, you saw cases in which dropping rows or columns associated with missing values would not be a good idea. There are other cases in which dropping rows or columns associated with missing values would be okay.

A few instances in which dropping a row might be okay are:

Dropping missing data associated with mechanical failures.

The missing data is in a column that you are interested in predicting.

Other cases when you should consider dropping data that are not associated with missing data:

Dropping columns with no variability in the data.

Dropping data associated with information that you know is not correct.

In handling removing data, you should think more about why is this missing or why is this data incorrectly input to see if an alternative solution might be used than dropping the values.


One common strategy for working with missing data is to understand the proportion of a column that is missing. If a large proportion of a column is missing data, this is a reason to consider dropping it.

There are easy ways using pandas to create dummy variables to track the missing values, so you can see if these missing values actually hold information (regardless of the proportion that are missing) before choosing to remove a full column

If an entire column or row is missing, we can remove it, as there is no information being provided.

The column with mixed heights, we should be able to (for the most part) map those to a consistent measurement (all meters or all feet). We don't want to just drop this.

If the response is missing, for those rows, we have nothing to predict. You might be interested in predicting those values. Without a target/response to predict, your model cannot learn. These rows are not providing information for training any sort of supervised learning model.

Though it is common to drop columns just because not many values exist, there may be value to grouping rows that have a column missing as compared to rows that do not have a missing value for that particular column.

If there is no variability (all the values are the same) in a column, it does not provide value for prediction or finding differences in your data. It should be dropped for this reason. Keeping it doesn't really hurt, but it can lead to confusing results as we will see later in this lesson.

When you have incorrect data, you do not want to input this information into your conclusions. You should attempt to correct these values, or you may need to drop them.


Drop any row with a missing value.

```python
all_drop  = small_dataset.dropna()
```


Drop only the row with all missing values.

```python
all_row = small_dataset.dropna(axis=0, how='all') #axis 0 specifies you drop, how all specifies that you 
```

Drop only the rows with missing values in column 3.

```python
only3_drop = small_dataset.dropna(subset=['col3'], how='any')
only3or1_drop = small_dataset.dropna(subset=['col1', 'col3'], how='any')
```

```python
X_2 = all_rm[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
y_2 = all_rm['Salary']

# Split data into training and test data, and fit a linear model
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2 , test_size=.30, random_state=42)
lm_2_model = LinearRegression(normalize=True)

# If our model works, it should just fit our model to the data. Otherwise, it will let us know.
try:
    lm_2_model.fit(X_2_train, y_2_train)
except:
    print("Oh no! It doesn't work!!!")


y_test_preds = lm_2_model.predict(X_2_test)# Predictions here
r2_test = r2_score(y_2_test, y_test_preds) # Rsquared here

# Print r2 to see result
r2_test

print("The number of salaries in the original dataframe is " + str(np.sum(df.Salary.notnull()))) 
print("The number of salaries predicted using our model is " + str(len(y_test_preds)))
print("This is bad because we only predicted " + str((len(y_test_preds))/np.sum(df.Salary.notnull())) + " of the salaries in the dataset.")
```

2. We can impute the missing values.


pro: you are not removing rows/cols associated with missing values

con: you are diluting the power of your features to predict well by reducing variability in those features


Imputation is likely the most common method for working with missing values for any data science team. The methods shown here included the frequently used methods of imputing the mean, median, or mode of a column into the missing values for the column.

There are many advanced techniques for imputing missing values including using machine learning and bayesian statistical approaches. This could be techniques as simple as using k-nearest neighbors to find the features that are most similar, and using the values those features have to fill in values that are missing or complex methods like those in the very popular AMELIA library.

Regardless your imputation approach, you should be very cautious of the BIAS you are imputing into any model that uses these imputed values. Though imputing values is very common, and often leads to better predictive power in machine learning models, it can lead to over generalizations. In extremely advanced techniques in Data Science, this can even mean ethical implications. Machines can only 'learn' from the data they are provided. If you provide biased data (due to imputation, poor data collection, etc.), it should be no surprise, you will achieve results that are biased

It is very common to impute in the following ways:

1). Impute the mean of a column.


```python
fill_mean = lambda col: col.fillna(col.mean())

try:
    new_df.apply(fill_mean, axis=0)
except:
    print('That broke...because column E is a string.')

new_df[['A', 'B', 'D']].apply(fill_mean, axis=0)
```

2). If you are working with categorical data or a variable with outliers, then use the mode of the column.


```python
fill_mode = lambda col: col.fillna(col.mode()[0])

new_df.apply(fill_mode, axis=0)
```


One of the main ways for working with categorical variables is using 0, 1 encodings. In this technique, you create a new column for every level of the categorical 
variable. The advantages of this approach include:

1). The ability to have differing influences of each level on the response.

2) You do not impose a rank of the categories.

3) The ability to interpret the results more easily than other encodings.

The disadvantages of this approach are that you introduce a large number of effects into your model. If you have a large number of categorical variables or categorical variables with a large number of levels, but not a large sample size, you might not be able to estimate the impact of each of these variables on your response variable. There are some rules of thumb that suggest 10 data points for each variable you add to your model. That is 10 rows for each column. This is a reasonable lower bound, but the larger your sample (assuming it is representative), the better.

Let's try out adding dummy variables for the categorical variables into the model. We will compare to see the improvement over the original model only using quantitative variables.


```python
df.dtypes
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

# dummy with nan: 'the NaNs are always encoded as 0'

pd.get_dummies(dummy_var_df['col1'])# Use this cell to write whatever code you need.


# dummy

dummy_cols_df = pd.get_dummies(dummy_var_df['col1'], dummy_na=True) #Create the three dummy columns for dummy_var_df

```

drop that col, dummy that col, concat

```python
def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for cat in cat_cols:
        try:
        	# Whether to get k-1 dummies out of k categorical levels by removing the first level.
            df = pd.concat([df.drop(cat,axis=1), pd.get_dummies(df[cat],prefix=col,prefix_sep = '_', drop_first =True, dummy_na = dummy_na) ],axis=1)
        #df[cat] = pd.get_dummies(df[cat], dummy_na = dummy_na)
        except:
            continue
    return df
```

```python
def clean_fit_linear_mod(df, response_col, cat_cols, dummy_na, test_size=.3, rand_state=42):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column 
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    
    Your function should:
    1. Drop the rows with missing response values
    2. Drop columns with NaN for all the values
    3. Use create_dummy_df to dummy categorical columns
    4. Fill the mean of the column for any missing values 
    5. Split your data into an X matrix and a response vector y
    6. Create training and test sets of data
    7. Instantiate a LinearRegression model with normalized data
    8. Fit your model to the training data
    9. Predict the response for the training data and the test data
    10. Obtain an rsquared value for both the training and test data
    '''
    # Drop the rows with missing response values
    df.dropna(subset = [response_col], axis = 0, inplace=True)
    
    # drop cols with all nans
    #allmis = df.columns[df.isnull().sum() == df.shape[0]]
    #df.dropna(subset = [allmis], how = 'any', inplace=True)
    df.dropna(how='all',axis=1,inplace=True)
    
    
    # dummy
    df = create_dummy_df(fill_df, cat_cols,dummy_na)
    
    # fill mean 
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)
    
    
    
    y = df[response_col]
    X = df.drop(response_col,axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = test_size, random_state = rand_state)
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train,y_train)
    pred2 = lm_model.predict(X_test)
    pred = lm_model.predict(X_train)
    test_score = r2_score(y_test,pred2)
    train_score = r2_score(y_train,pred)
    
    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test


#Test your function with the above dataset
test_score, train_score, lm_model, X_train, X_test, y_train, y_test = clean_fit_linear_mod(df_new, 'Salary', cat_cols_lst, dummy_na=False)

```

https://pbpython.com/categorical-encoding.html



```python

Signature: t.find_optimal_lm_mod(X, y, cutoffs, test_size=0.3, random_state=42, plot=True)
Source:   
def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test
```


```python
def clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    # Drop rows with missing salary values
    df = df.dropna(subset=['Salary'], axis=0)
    y = df['Salary']
    
    #Drop respondent and expected salary columns
    df = df.drop(['Respondent', 'ExpectedSalary', 'Salary'], axis=1)
    
    # Fill numeric columns with the mean
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)
        
    # Dummy the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
    X = df
    return X, y
    
#Use the function to create X and y
X, y = clean_data(df)
```



```python
#cutoffs here pertains to the number of missing values allowed in the used columns.
#Therefore, lower values for the cutoff provides more predictors in the model.
cutoffs = [5000, 3500, 2500, 1000, 100, 50, 30, 25]

#Run this cell to pass your X and y to the model for testing
r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = t.find_optimal_lm_mod(X, y, cutoffs)
```


The default penalty on coefficients using linear regression in sklearn is a ridge (also known as an L2) penalty. Because of this penalty, and that all the variables were normalized, we can look at the size of the coefficients in the model as an indication of the impact of each variable on the salary. The larger the coefficient, the larger the expected impact on salary.

Use the space below to take a look at the coefficients. Then use the results to provide the True or False statements based on the data.

```python
def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

#Use the function
coef_df = coef_weights(lm_model.coef_, X_train)

#A quick look at the top results
coef_df.head(20)
```


3). Impute 0, a very small number, or a very large number to differentiate missing values from other values.


4). Use knn to impute values based on features that are most similar.


In general, you should try to be more careful with missing data in understanding the real world implications and reasons for why the missing values exist. At the same time, these solutions are very quick, and they enable you to get models off the ground. You can then iterate on your feature engineering to be more careful as time permits.



3. We can build models that work around them, and only use the information provided.




Two techniques for deploying your results include:

1) Automated techniques built into computer systems or across the web. You will do this later in this program!

2) Communicate results with text, images, slides, dashboards, or other presentation methods to company stakeholders.
To get some practice with this second technique, you will be writing a blog post for the first project and turning in a Github repository that shares your work.

As a data scientist, communication of your results to both other team members and to less technical members of a company is a critical component