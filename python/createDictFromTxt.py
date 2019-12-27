https://www.tutorialspoint.com/How-to-create-a-Python-dictionary-from-text-file



```python
"""
Assuming a following text file (dict.txt) is present

1 aaa
2 bbb
3 ccc

Following Python code reads the file using open() function. Each line as string is split at space character. First component is used as key and second as value
"""
d = {}
with open("dict.txt") as f:
  for line in f:
    (key,val) = line.split()
    d[int(key)] = val

return d
```

another example

```python
import re

def str2dict(filename="temp.txt"):
    results = {}
    with open(filename, "r") as cache:
        # read file into a list of lines
        lines = cache.readlines()
        # loop through lines
        for line in lines:
            # skip lines starting with "--".
            if not line.startswith("--"):
                # replace random amount of spaces (\s) with tab (\t),
                # strip the trailing return (\n), split into list using
                # "\t" as the split pattern
                line = re.sub("\s\s+", "\t", line).strip().split("\t")
                # use first item in list for the key, join remaining list items
                # with ", " for the value.
                results[line[0]] = ", ".join(line[1:])

    return results

print (str2dict("temp.txt"))
```

readlines(): turn file into list of lines
  
strip(): removes characters (usually space/enter stuff) from both left and right based on the argument 

then assess words in line by position: e,g line[0]
