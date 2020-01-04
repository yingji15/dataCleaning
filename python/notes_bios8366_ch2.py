

# read in chunks

data_chunks = pd.read_csv("../data/microbiome.csv", chunksize=14)

mean_tissue = pd.Series({chunk.iloc[0].Taxon:chunk.Tissue.mean() for chunk in data_chunks})
    
mean_tissue



# missing

pd.read_csv("../data/microbiome_missing.csv", na_values=['?', -99999]).head(20)

# read excel
mb = pd.read_excel('../data/microbiome/MID2.xls', sheetname='Sheet 1', header=None)
mb.head()

# limit show

pd.set_option('display.max_rows', 10)


# create new id by combine columns

player_id = baseball.player + baseball.year.astype(str)
baseball_newind = baseball.copy()
baseball_newind.index = player_id
baseball_newind

# check unique
baseball_newind.index.is_unique

# **Reindexing** allows users to manipulate the data labels in a DataFrame.

# reverse index, A simple use of `reindex` is to alter the order of the rows:



baseball.reindex(baseball.index[::-1]).head()

# missing

# Missing values can be filled as desired, either with selected values, or by rule:


baseball.reindex(id_range, method='ffill').head()


baseball.reindex(id_range, fill_value='charliebrown', columns=['player']).head()



foo.isnull()
# `dropna` drops entire rows in which one or more values are missing.
test_scores.dropna()

test_scores[test_scores.notnull()]


# This can be overridden by passing the `how='all'` argument, which only drops a row when every field is a missing value.



test_scores.dropna(how='all')


# This can be customized further by specifying how many values need to be present before a row is dropped via the `thresh` argument.



test_scores.dropna(thresh=10)

# We can alter values in-place using `inplace=True`.



test_scores.prev_disab.fillna(0, inplace=True)


# Missing values can also be interpolated, using any one of a variety of methods:



test_scores.fillna(method='bfill')

# remove rows or columns via the `drop` method

baseball.drop(['ibb','hbp'], axis=1)

# select rows

baseball_newind[baseball_newind.ab>500]


# For a more concise (and readable) syntax, we can use the new `query` method to perform selection on a `DataFrame`. 
#Instead of having to type the fully-specified column, we can simply pass a string that describes what to select. The query above is then simply:



baseball_newind.query('ab > 500')


# The indexing field `loc` allows us to select subsets of rows and columns in an intuitive way:


baseball_newind.loc['gonzalu01ARI2006', ['h','X2b', 'X3b', 'hr']]

# We can also apply functions to each column or row of a `DataFrame`

# In[37]:


import numpy as np

stats.apply(np.median)



def range_calc(x):
    return x.max() - x.min()



stat_range = lambda x: x.max() - x.min()
stats.apply(stat_range)


def slugging(x): 
    bases = x['h']-x['X2b']-x['X3b']-x['hr'] + 2*x['X2b'] + 3*x['X3b'] + 4*x['hr']
    ab = x['ab']+1e-6
    
    return bases/ab

baseball.apply(slugging, axis=1).round(3)

# ## Sorting and Ranking

# default order: ascending, rows
baseball_newind.sort_index(ascending=False).head()
# over cols
baseball_newind.sort_index(axis=1).head()

#We can also use `sort_values` to sort a `Series` by value, rather than by label.


baseball.hr.sort_values()

baseball[['player','sb','cs']].sort_values(ascending=[False,True], 
                                           by=['sb', 'cs']).head(10)
                                           **Ranking** does not re-arrange data, but instead returns an index that ranks each value relative to others in the Series.

#rank: doesn't re-order data


baseball.hr.rank()

# break ties by first occurence

baseball.hr.rank(method='first')

baseball.rank(ascending=False).head()




baseball[['r','h','hr']].rank(ascending=False).head()


# summary

extra_bases = baseball[['X2b','X3b','hr']].sum(axis=1)
extra_bases.sort_values(ascending=False)

baseball.describe()

baseball.hr.cov(baseball.X2b)

#If we have a `DataFrame` with a hierarchical index (or indices), summary statistics can be applied with respect to any of the index levels:



mb.sum(level='Taxon')

# date time

from datetime import datetime

from datetime import date, time

my_age = now - datetime(1970, 9, 3)
my_age


segments.seg_length.hist(bins=500)

# Though most of the transits appear to be short, there are a few longer distances that make the plot difficult to read. 

# This is where a transformation is useful:

segments.seg_length.apply(np.log).hist(bins=500)

# The `strptime` method parses a string representation of a date and/or time field, 
# according to the expected format of this information.


datetime.strptime(segments.st_time.loc[0], '%m/%d/%y %H:%M')

# The `dateutil` package includes a parser that attempts to detect the format of the date strings, 
# and convert them automatically.

# In[16]:


from dateutil.parser import parse





parse(segments.st_time.loc[0])


# We can convert all the dates in a particular column by using the `apply` method.




segments.st_time.apply(lambda d: datetime.strptime(d, '%m/%d/%y %H:%M'))

# As a convenience, Pandas has a `to_datetime` method that will parse and convert an entire Series of formatted strings into `datetime` objects.


pd.to_datetime(segments.st_time[:10])


# The `read_*` functions now have an optional `parse_dates` argument that try to convert any columns passed to it into `datetime` format upon import:



segments = pd.read_csv("../data/AIS/transit_segments.csv", parse_dates=['st_time', 'end_time'])


# Columns of the `datetime` type have an **accessor** to easily extract properties of the data type. 
# This will return a `Series`, with the same row index as the `DataFrame`. For example:




segments.st_time.dt.month.head()



segments.st_time.dt.hour.head()


#  easily filter rows by particular temporal attributes:

segments[segments.st_time.dt.month==2].head()

# time zone

segments.st_time.dt.tz_localize('UTC').head()


# convert

segments.st_time.dt.tz_localize('UTC').dt.tz_convert('US/Eastern').head()

# itertools:product cartesian product, equivalent to a nested for-loop
# simulate data with separate columns of date time

from itertools import product

years = range(2000, 2018)
months = range(1, 13)
days = range(1, 29)
hours = range(24)

temp_df = pd.DataFrame(list(product(years, months, days, hours)), 
                         columns=['year', 'month', 'day', 'hour'])

#
temp_df.index = pd.to_datetime(temp_df[['year', 'month', 'day', 'hour']])

##########################################
# Merging and joining DataFrame objects
###########################################


# The table of vessel information has a one-to-many relationship with the segments.
vessels.type.value_counts()

# In Pandas, we can combine tables according to the value of one or more keys that are used to identify rows, 
#much like an index. Using a trivial example:
df1 = pd.DataFrame(dict(id=range(4), age=np.random.randint(18, 31, size=4)))
df2 = pd.DataFrame(dict(id=list(range(3))+list(range(3)), 
                        score=np.random.random(size=6)))
                        
#The outer join above yields the union of the two tables, so all rows are represented, with missing values inserted as appropriate                        
pd.merge(df1, df2, how='outer')

vessels.merge(segments, left_index=True, right_on='mmsi').head()

## Concatenation
# 
# A common data manipulation is appending rows or columns to a dataset that already conform to the dimensions of the exsiting rows or colums, respectively. 

#In NumPy, this is done either with `concatenate` or the convenience "functions" `c_` and `r_`:



np.concatenate([np.random.random(5), np.random.random(5)])

# by row
np.r_[np.random.random(5), np.random.random(5)]

# by col 
np.c_[np.random.random(5), np.random.random(5)]

# With Pandas' indexed data structures, overlap in index values between two data structures affects how they are concatenate.

mb1 = pd.read_excel('../data/microbiome/MID1.xls', 'Sheet 1', index_col=0, header=None)
mb2 = pd.read_excel('../data/microbiome/MID2.xls', 'Sheet 1', index_col=0, header=None)
mb1.shape, mb2.shape



mb1.head()


# Let's give the index and columns meaningful labels:



mb1.columns = mb2.columns = ['Count']



mb1.index.name = mb2.index.name = 'Taxon'

# `axis=0` (the default), we will obtain another data frame with the the rows concatenated:



pd.concat([mb1, mb2], axis=0).shape

pd.concat([mb1, mb2], axis=0).index.is_unique

# Concatenating along `axis=1` will concatenate column-wise, but respecting the indices of the two DataFrames.


pd.concat([mb1, mb2], axis=1).shape

# taxa in both
pd.concat([mb1, mb2], axis=1, join='inner').head()


# use the second table to fill values absent from the first table, we could use combine_first

mb1.combine_first(mb2).head()


# We can also create a hierarchical index based on keys identifying the original tables.



pd.concat([mb1, mb2], keys=['patient1', 'patient2']).head()

# Alternatively, you can pass keys to the concatenation by supplying the DataFrames (or Series) as a dict, 
# resulting in a "wide" format table.

# In[64]:


pd.concat(dict(patient1=mb1, patient2=mb2), axis=1).head()

# exercise: import all files within dir and concat into df
# sol 1 

import glob

path = r'C:\DRO\DCL_rawdata_files' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

#sol2: more concise

path = r'C:\DRO\DCL_rawdata_files'                     # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
# doesn't create a list, nor does it append to one

#####################
# longitudinal: long vs wide
############################

# This illustrates the two formats for longitudinal data: long and wide formats.
# This dataset includes repeated measurements of the same individuals (longitudinal data).



# The `stack` method rotates the data frame so that columns are represented in rows:
stacked = cdystonia.stack()
stacked

# To complement this, `unstack` pivots from rows back to columns.

# In[68]:


stacked.unstack().head()

# For this dataset, it makes sense to create a hierarchical index based on the patient and observation:

# In[69]:


cdystonia2 = cdystonia.set_index(['patient','obs'])

# If we want to transform this data so that repeated measurements are in columns, 
# we can `unstack` the `twstrs` measurements according to `obs`.


# cols: obs 1,2,3...
# rows: patient
# fill: twstrs val
twstrs_wide = cdystonia2['twstrs'].unstack('obs')
twstrs_wide.head()

cdystonia_wide = (cdystonia[['patient','site','id','treat','age','sex']]
                  .drop_duplicates()
                  .merge(twstrs_wide, right_index=True, left_on='patient', how='inner')
                  .head())
                  
                  

# # To convert our "wide" format back to long


# leaving just two non-identifier columns, a *variable* and its corresponding *value*, which can both be renamed using optional arguments.


pd.melt(cdystonia_wide, id_vars=['patient','site','id','treat','age','sex'], 
        var_name='obs', value_name='twsters').head()
        
        
# Its typically better to store data in long format because additional data can be included as additional rows in the database, 
# while wide format requires that the entire database schema be altered by adding columns to every row as data are collected.

## Pivoting
# 
# The `pivot` method allows a DataFrame to be transformed easily between long and wide formats in the same way as a pivot table is created in a spreadsheet. 
# It takes three arguments: `index`, `columns` and `values`, 
# index: corresponding to the DataFrame index (the row headers), columns and cell values, respectively.
# 
# For example, we may want the `twstrs` variable (the response variable) in wide format according to patient, 
# as we saw with the unstacking method above:




cdystonia.pivot(index='patient', columns='obs', values='twstrs').head()

# #and allows the values of the table to be populated using an arbitrary aggregation function
cdystonia.pivot_table(index=['site', 'treat'], columns='week', values='twstrs', 
                      aggfunc=max).head(20)
                      
# For a simple cross-tabulation of group frequencies, the `crosstab` function (not a method) aggregates counts of data according to factors in rows and columns. The factors may be hierarchical if desired.



pd.crosstab(cdystonia.sex, cdystonia.site)


###################
# method chaining 
#################

# The goal is to summarize this data by age groups and bi-weekly period, so that we can see how the outbreak affected different ages over the course of the outbreak.

pd.cut(measles.YEAR_AGE, [0,5,10,15,20,25,30,35,40,100], right=False)


# What we then want is the number of occurences in each combination, which we can obtain by checking the `size` of each grouping:



(measles.assign(AGE_GROUP=pd.cut(measles.YEAR_AGE, [0,5,10,15,20,25,30,35,40,100], right=False))
                        .groupby(['ONSET', 'AGE_GROUP'])
                        .size()).head(10)
                        
                        
# This results in a hierarchically-indexed `Series`, which we can pivot into a `DataFrame` by simply unstacking:

# In[84]:


(measles.assign(AGE_GROUP=pd.cut(measles.YEAR_AGE, [0,5,10,15,20,25,30,35,40,100], right=False))
                        .groupby(['ONSET', 'AGE_GROUP'])
                        .size()
                        .unstack()).head(5)
# Now, fill replace the missing values with zeros:

# In[85]:


(measles.assign(AGE_GROUP=pd.cut(measles.YEAR_AGE, [0,5,10,15,20,25,30,35,40,100], right=False))
                        .groupby(['ONSET', 'AGE_GROUP'])
                        .size()
                        .unstack()
                        .fillna(0)).head(5)

# Finally, we want the counts in 2-week intervals, rather than as irregularly-reported days, which yields our the table of interest:

# In[86]:


case_counts_2w = (measles.assign(AGE_GROUP=pd.cut(measles.YEAR_AGE, [0,5,10,15,20,25,30,35,40,100], right=False))
                        .groupby(['ONSET', 'AGE_GROUP'])
                        .size()
                        .unstack()
                        .fillna(0)
                        .resample('2W')
                        .sum())

case_counts_2w


# From this, it is easy to create meaningful plots and conduct analyses:

# In[87]:


case_counts_2w.plot(cmap='magma');


# The pandas `pipe` DataFrame method allows users to apply a function to a DataFrame, as if it were a method. The lone restriction on the function is that it must return the modified DataFrame as its return value.


def to_proportions(df, axis=1):
    row_totals = df.sum(axis)
    return df.div(row_totals, True - axis)


# We can then use the `pipe` method in our chain, with the function as its argument:

# In[89]:


case_prop_2w = (measles.assign(AGE_GROUP=pd.cut(measles.YEAR_AGE, [0,5,10,15,20,25,30,35,40,100], right=False))
                        .groupby(['ONSET', 'AGE_GROUP'])
                        .size()
                        .unstack()
                        .fillna(0)
                        .resample('2W')
                        .sum()
                        .pipe(to_proportions))

case_prop_2w
############################
## Data transformation                        
#############################


# Dealing with duplicates

vessels.duplicated(subset='names')


# In[92]:


vessels.drop_duplicates(['names'])

# Value replacement
# 
# Frequently, we get data columns that are encoded as strings that we wish to represent numerically for the purposes of including it in a quantitative analysis. For example, consider the treatment variable in the cervical dystonia dataset:

# In[93]:


cdystonia.treat.value_counts()


# A logical way to specify these numerically is to change them to integer values, perhaps using "Placebo" as a baseline value. If we create a dict with the original values as keys and the replacements as values, we can pass it to the `map` method to implement the changes.

# In[94]:


treatment_map = {'Placebo': 0, '5000U': 1, '10000U': 2}


# In[95]:


cdystonia['treatment'] = cdystonia.treat.map(treatment_map)
cdystonia.treatment

# if we simply want to replace particular values in a `Series` or `DataFrame`

vals = pd.Series([float(i)**10 for i in range(10)])
vals


# In[97]:


np.log(vals)


# In such situations, we can replace the zero with a value so small that it makes no difference to the ensuing analysis. We can do this with `replace`.


vals = vals.replace(0, 1e-6)
np.log(vals)


# We can also perform the same replacement that we used `map` for with `replace`:



cdystonia2.treat.replace({'Placebo': 0, '5000U': 1, '10000U': 2})


# Inidcator variables

#The Pandas function `get_dummies` (indicator variables are also known as *dummy variables*) makes this transformation straightforward.


top5 = vessels.type.isin(vessels.type.value_counts().index[:5])
top5.head(10)





vessels5 = vessels[top5]


pd.get_dummies(vessels5.type).head(10)




# Categorical Data

# We can convert this to a `category` type either by the `Categorical` constructor, or casting the column using `astype`:

# In[104]:


pd.Categorical(cdystonia.treat)


# In[105]:


cdystonia['treat'] = cdystonia.treat.astype('category')


# In[106]:


cdystonia.treat.describe()


# However, an ordering can be imposed. The order is lexical by default, but will assume the order of the listed categories to be the desired order.



cdystonia.treat.cat.categories = ['Placebo', '5000U', '10000U']



cdystonia.treat.cat.as_ordered().head()



# Permutation and sampling

# For some data analysis tasks, such as simulation, we need to be able to randomly reorder our data, or draw random values from it. Calling NumPy's `permutation` function with the length of the sequence you want to permute generates an array with a permuted sequence of integers, which can be used to re-order the sequence.

# In[114]:


new_order = np.random.permutation(len(segments))


# For random sampling, `DataFrame` and `Series` objects have a `sample` method that can be used to draw samples, with or without replacement:

# In[117]:


vessels.sample(n=10)


# In[118]:


vessels.sample(n=10, replace=True)



# Data aggregation and GroupBy operations

# * **aggregation**, such as computing the sum of mean of each group, which involves applying a function to each group and returning the aggregated results
# * **slicing** the DataFrame into groups and then doing something with the resulting slices (*e.g.* plotting)
# * group-wise **transformation**, such as standardization/normalization




cdystonia_grouped = cdystonia.groupby(cdystonia.patient)

cdystonia_grouped.mean().add_suffix('_mean').head()


# In[125]:


# The median of the `twstrs` variable
cdystonia_grouped['twstrs'].quantile(0.5)


# If we wish, we can easily aggregate according to multiple keys:

# In[126]:


cdystonia.groupby(['week','site']).mean().head()

# Alternately, we can **transform** the data, using a function of our choice with the `transform` method:

# In[127]:


normalize = lambda x: (x - x.mean())/x.std()

cdystonia_grouped.transform(normalize).head()

# It is easy to do column selection within `groupby` operations, 
# if we are only interested split-apply-combine operations on a subset of columns:

# In[128]:


cdystonia_grouped['twstrs'].mean().head()


# In[129]:


# This gives the same result as a DataFrame
cdystonia_grouped[['twstrs']].mean().head()

cdystonia2.groupby(level='obs', axis=0)['twstrs'].mean()

# Apply
# 
# We can generalize the split-apply-combine methodology by using `apply` function. This allows us to invoke any function we wish on a grouped dataset and recombine them into a DataFrame.

# The function below takes a DataFrame and a column name, sorts by the column, and takes the `n` largest values of that column. We can use this with `apply` to return the largest values from every group in a DataFrame in a single call. 

# In[134]:


def top(df, column, n=5):
    return df.sort_values(by=column, ascending=False)[:n]


# Say we wanted to return the 3 longest segments travelled by each ship:

# In[135]:


top3segments = segments_merged.groupby('mmsi').apply(top, column='seg_length', n=3)[['names', 'seg_length']]

# Using the string methods `split` and `join` we can create an index that just uses the first three classifications: domain, phylum and class.

# In[137]:


class_index = mb1.index.map(lambda x: ' '.join(x.split(' ')[:3]))


# In[138]:


mb_class = mb1.copy()
mb_class.index = class_index


# We can re-establish a unique index by summing all rows with the same class, using `groupby`:

# In[140]:


mb_class.groupby(level=0).sum().head(10)        
        
