https://www.tutorialspoint.com/How-to-create-a-Python-dictionary-from-text-file

"""
Assuming a following text file (dict.txt) is present

1 aaa
2 bbb
3 ccc

Following Python code reads the file using open() function. Each line as string is split at space character. First component is used as key and second as value
"""

```python
d = {}
with open("dict.txt") as f:
  for line in f:
    (key,val) = line.split()
    d[int(key)] = val

return d
```
