```
df.loc[1905:, 'Date of Publication'].head(10)
Identifier
1905           1888
1929    1839, 38-54
2836        [1897?]
2854           1865
2956        1860-63
2957           1873
3017           1866
3131           1899
4598           1814
4884           1820
Name: Date of Publication, dtype: object
```

A particular book can have only one date of publication. Therefore, we need to do the following:

Remove the extra dates in square brackets, wherever present: 1879 [1878]

Convert date ranges to their “start date”, wherever present: 1860-63; 1839, 38-54

Completely remove the dates we are not certain about and replace them with NumPy’s NaN: [1897?]

Convert the string nan to NumPy’s NaN value

Synthesizing these patterns, we can actually take advantage of a single regular expression to extract the publication year:

```
regex = r'^(\d{4})'
```

r: meant to find any four digits at the beginning of a string, which suffices for our case.

(): denote a capturing group

^: start with

\d: any digit

{4}: repeat 4 times

```python
>>> extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
>>> extr.head()
Identifier
206    1879
216    1868
218    1869
472    1851
480    1857
Name: Date of Publication, dtype: object

df['Date of Publication'] = pd.to_numeric(extr)

df['Date of Publication'].isnull().sum() / len(df)

```

```
np.where(condition, then, else)
```

Here, condition is either an array-like object or a boolean mask. then is the value to be used if condition evaluates to True, and else is the value to be used otherwise.

```python
df['Place of Publication'] = np.where(london, 'London',
                                      np.where(oxford, 'Oxford',
                                               pub.str.replace('-', ' ')))
```

