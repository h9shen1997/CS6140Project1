# CS6140 Project 1

## Description of project
My project is revolving around the diamond price dataset, where the price of the diamond is determined by 
multiple factors, such as carat, cut, color, clarity, depth and table size. I used PCA and several regression
methods to find the correlations between these independent variables to the pricing of diamond.

## Operating system & IDE
I run my project in PyCharm on macOS.

## Instruction for running mu executable programs
```
cd <project_folder>
# the first argument is the csv file name and the second argument is the test data percentage
python3 main.py diamonds.csv 25
```

## Instruction for extension testing
Extension 1:

To test the data organization functionality, you can change the random seed in the split_data function and see if the 
result is any different. I also have attached some sample run result in the report.

Once you change the seed to another value, just use the same command to run.

Extension 2:

Ridge and Lasso contrast with different alpha values and comparison with the original multiple linear regression 
regarding their coefficients change.

Just run the code using the same command to run. You can also view the code in the Extension section in the main.py.

## Time travel days
No time travel day is used
