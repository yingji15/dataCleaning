# l2. udacity software engineering

How this Course is Organized

Software Engineering Practices Part 1 covers how to write well documented, modularized code.

Software Engineering Practices Part 2 discusses testing your code and logging.

Introduction to Object-Oriented Programming gives you an overview of this programming style and prepares you to write your own Python package.

Introduction to Web Development covers building a web application data dashboard.

# Software Engineering Practices Part 1

## Writing clean and modular code

PRODUCTION CODE: software running on production servers to handle live users and data of the intended audience. Note this is different from production quality code, which describes code that meets expectations in reliability, efficiency, etc., for production. Ideally, all code in production meets these expectations, but this is not always the case.

CLEAN: readable, simple, and concise. A characteristic of production quality code that is crucial for collaboration and maintainability in software development.

MODULAR: logically broken up into functions and modules. Also an important characteristic of production quality code that makes your code more organized, efficient, and reusable.

MODULE: a file. Modules allow code to be reused by encapsulating them into files that can be imported into other files.

Refactoring Code

REFACTORING: restructuring your code to improve its internal structure, without changing its external functionality. This gives you a chance to clean and modularize your program after you've got it working.

Since it isn't easy to write your best code while you're still trying to just get it working, allocating time to do this is essential to producing high quality code. Despite the initial time and effort required, this really pays off by speeding up your development time in the long run.
You become a much stronger programmer when you're constantly looking to improve your code. The more you refactor, the easier it will be to structure and write good code the first time.

## Writing efficient code

1) Use meaningful names

Be descriptive and imply type - E.g. for booleans, you can prefix with is_ or has_ to make it clear it is a condition. You can also use part of speech to imply types, like verbs for functions and nouns for variables.

Be consistent but clearly differentiate - E.g. age_list and age is easier to differentiate than ages and age.

Avoid abbreviations and especially single letters - (Exception: counters and common math variables) Choosing when these exceptions can be made can be determined based on the audience for your code. If you work with other data scientists, certain variables may be common knowledge. While if you work with full stack engineers, it might be necessary to provide more descriptive names in these cases as well.

Long names != descriptive names - You should be descriptive, but only with relevant information. E.g. good functions names describe what they do well without including details about implementation or highly specific uses.

2) Tip: Use whitespace properly

Organize your code with consistent indentation - the standard is to use 4 spaces for each indent. You can make this a default in your text editor.

Separate sections with blank lines to keep your code well organized and readable.

Try to limit your lines to around 79 characters, which is the guideline given in the PEP 8 style guide. In many good text editors, there is a setting to display a subtle line that indicates where the 79 character limit is.


3) Writing Modular Code

* Tip: DRY (Don't Repeat Yourself)

Don't repeat yourself! Modularization allows you to reuse parts of your code. Generalize and consolidate repeated code in functions or loops.

* Tip: Abstract out logic to improve readability

Abstracting out code into a function not only makes it less repetitive, but also improves readability with descriptive function names. Although your code can become more readable when you abstract out logic into functions, it is possible to over-engineer this and have way too many modules, so use your judgement.

* Tip: Minimize the number of entities (functions, classes, modules, etc.)

There are tradeoffs to having function calls instead of inline logic. If you have broken up your code into an unnecessary amount of functions and modules, you'll have to jump around everywhere if you want to view the implementation details for something that may be too small to be worth it. Creating more modules doesn't necessarily result in effective modularization.

* Tip: Functions should do one thing

Each function you write should be focused on doing one thing. If a function is doing multiple things, it becomes more difficult to generalize and reuse. Generally, if there's an "and" in your function name, consider refactoring.

* Tip: Arbitrary variable names can be more effective in certain functions

Arbitrary variable names in general functions can actually make the code more readable.

* Tip: Try to use fewer than three arguments per function

Try to use no more than three arguments when possible. This is not a hard rule and there are times it is more appropriate to use many parameters. But in many cases, it's more effective to use fewer arguments. Remember we are modularizing to simplify our code and make it more efficient to work with. If your function has a lot of parameters, you may want to rethink how you are splitting this up.


```python
new_df = df.rename(columns={'fixed acidity': 'fixed_acidity',
                             'volatile acidity': 'volatile_acidity',
                             'citric acid': 'citric_acid',
                             'residual sugar': 'residual_sugar',
                             'free sulfur dioxide': 'free_sulfur_dioxide',
                             'total sulfur dioxide': 'total_sulfur_dioxide'
                            })
new_df.head()

# a better way
labels = list(df.columns)
labels[0] = labels[0].replace(' ', '_')

# my code
rename = lambda x: x.replace(' ','_')
    
list(map(rename,df.columns))


df.columns = [label.replace(' ', '_') for label in df.columns]
df.head()

def fea_quality(df,x):
    med = df[x].median()
    for index, value in enumerate(df.x):
        if value > med:
            df.loc[index,'x'] = 'high'
        else:
            df.loc[index,'x'] = 'low'
 
for feature in df.columns[:-1]:
	fea_quality(df,feature)
	print(df.groupby(feature).quality.mean(),'\n')
```

## Efficient Code

Knowing how to write code that runs efficiently is another essential skill in software development. Optimizing code to be more efficient can mean making it:

Execute faster

Take up less space in memory/storage

* vector operations

* data structure

```python
import time
import pandas as pd
import numpy as np


with open('books_published_last_two_years.txt') as f:
    recent_books = f.read().split('\n')
    
with open('all_coding_books.txt') as f:
    coding_books = f.read().split('\n')

start = time.time()
recent_coding_books =  np.intersect1d(recent_books, coding_books) # TODO: compute intersection of lists
print(len(recent_coding_books))
print('Duration: {} seconds'.format(time.time() - start))


start = time.time()
recent_coding_books = set(recent_books).intersection(coding_books)
#another way
recent_coding_books =  set.intersection(set(recent_books), set(coding_books))# TODO: compute intersection of lists
print(len(recent_coding_books))
print('Duration: {} seconds'.format(time.time() - start))
```


## Code refactoring

## Adding meaningful documentation

DOCUMENTATION: additional text or illustrated information that comes with or is embedded in the code of software.
Helpful for clarifying complex parts of code, making your code easier to navigate, and quickly conveying how and why different components of your program are used.

Several types of documentation can be added at different levels of your program:

* In-line Comments - line level

In-line comments are text following hash symbols throughout your code. They are used to explain parts of your code, and really help future contributors understand your work.

One way comments are used is to document the major steps of complex code to help readers follow. Then, you may not have to understand the code to follow what it does. However, others would argue that this is using comments to justify bad code, and that if code requires comments to follow, it is a sign refactoring is needed.

Comments are valuable for explaining where code cannot. For example, the history behind why a certain method was implemented a specific way. Sometimes an unconventional or seemingly arbitrary approach may be applied because of some obscure external variable causing side effects. These things are difficult to explain with code.

* Docstrings - module and function level

https://www.python.org/dev/peps/pep-0257/

* Project Documentation - project level

## Using version control


* Scenario #1

Let's walk through the git commands that go along with each step in the scenario you just observed in the video above.

STEP 1: You have a local version of this repository on your laptop, and to get the latest stable version, you pull from the develop branch.
Switch to the develop branch
git checkout develop

Pull latest changes in the develop branch

```
git pull
```

STEP 2: When you start working on this demographic feature, you create a new branch for this called demographic, and start working on your code in this branch.
Create and switch to new branch called demographic from develop branch

```
git checkout -b demographic
```

Work on this new feature and commit as you go

```
git commit -m 'added gender recommendations'
git commit -m 'added location specific recommendations'
```

...

STEP 3: However, in the middle of your work, you need to work on another feature. So you commit your changes on this demographic branch, and switch back to the develop branch.

Commit changes before switching

```
git commit -m 'refactored demographic gender and location recommendations '
```

Switch to the develop branch

```
git checkout develop
```


STEP 4: From this stable develop branch, you create another branch for a new feature called friend_groups.
Create and switch to new branch called friend_groups from develop branch

```
git checkout -b friend_groups
```

STEP 5: After you finish your work on the friend_groups branch, you commit your changes, switch back to the development branch, merge it back to the develop branch, and push this to the remote repository’s develop branch.
Commit changes before switching


```
git commit -m 'finalized friend_groups recommendations '
```

Switch to the develop branch

```
git checkout develop
```

Merge friend_groups branch to develop

```
git merge --no-ff friends_groups
```

Push to remote repository

```
git push origin develop
```

STEP 6: Now, you can switch back to the demographic branch to continue your progress on that feature.
Switch to the demographic branch

```
git checkout demographic
``


* Scenario #2

Let's walk through the git commands that go along with each step in the scenario you just observed in the video above.

Step 1: You check your commit history, seeing messages of the changes you made and how well it performed.
View log history

```
git log
```

Step 2: The model at this commit seemed to score the highest, so you decide to take a look.
Checkout a commit

```
git checkout bc
```

After inspecting your code, you realize what modifications made this perform well, and use those for your model.

Step 3: Now, you’re pretty confident merging this back into the development branch, and pushing the updated recommendation engine.

Switch to develop branch

```
git checkout develop
```

Merge friend_groups branch to develop

```
git merge --no-ff friend_groups
```

Push changes to remote repository

```
git push origin develop
```


* Scenario #3

Let's walk through the git commands that go along with each step in the scenario you just observed in the video above.

Step 1: Andrew commits his changes to the documentation branch, switches to the development branch, and pulls down the latest changes from the cloud on this development branch, including the change I merged previously for the friends group feature.

Commit changes on documentation branch

```
git commit -m "standardized all docstrings in process.py"
```

Switch to develop branch

```
git checkout develop
```

Pull latest changes on develop down

```
git pull
```

Step 2: Then, Andrew merges his documentation branch on the develop branch on his local repository, and then pushes his changes up to update the develop branch on the remote repository.
Merge documentation branch to develop

```
git merge --no-ff documentation
```

Push changes up to remote repository

```
git push origin develop
```

Step 3: After the team reviewed both of your work, they merge the updates from the development branch to the master branch. Now they push the changes to the master branch on the remote repository. These changes are now in production.

Merge develop to master

```
git merge --no-ff develop
```

Push changes up to remote repository

```
git push origin master
```

Resources

There's a great article on a successful git branching strategy that you should really read here.

Note on Merge Conflicts


For the most part, git makes merging changes between branches really simple. However, there are some cases where git will be confused on how to combine two changes, and asks you for help. This is called a merge conflict.

Mostly commonly, this happens when two branches modify the same file.

For example, in this situation, let’s say I deleted a line that Andrew modified on his branch. Git wouldn’t know whether to delete the line or modify it. Here, you need to tell git which change to take, and some tools even allow you to edit the change manually. If it isn’t straightforward, you may have to consult with the developer of the other branch to handle a merge conflict.


Model Versioning

In the previous example, you may have noticed that each commit was documented with a score for that model. This is one simple way to help you keep track of model versions. Version control in data science can be tricky, because there are many pieces involved that can be hard to track, such as large amounts of data, model versions, seeds, hyperparameters, etc.

Here are some resources for useful ways and tools for managing versions of models and large data. These are here for you to explore, but are not necessary to know now as you start your journey as a data scientist. On the job, you’ll always be learning new skills, and many of them will be specific to the processes set in your company.

https://algorithmia.com/blog/how-to-version-control-your-production-machine-learning-models