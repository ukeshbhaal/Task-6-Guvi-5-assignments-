# Task-6-Guvi-5-assignments-
Assignment8


[ ]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


How Much is Your Car Worth?
Data about the retail price of 2005 General Motors cars can be found in car_data.csv.

The columns are:

Price: suggested retail price of the used 2005 GM car in excellent condition.
Mileage: number of miles the car has been driven
Make: manufacturer of the car such as Saturn, Pontiac, and Chevrolet
Model: specific models for each car manufacturer such as Ion, Vibe, Cavalier
Trim (of car): specific type of car model such as SE Sedan 4D, Quad Coupe 2D
Type: body type such as sedan, coupe, etc.
Cylinder: number of cylinders in the engine
Liter: a more specific measure of engine size
Doors: number of doors
Cruise: indicator variable representing whether the car has cruise control (1 = cruise)
Sound: indicator variable representing whether the car has upgraded speakers (1 = upgraded)
Leather: indicator variable representing whether the car has leather seats (1 = leather)


Tasks, Part 1
Find the linear regression equation for mileage vs price.
Chart the original data and the equation on the chart.
Find the equation's R2 score (use the .score method) to determine whether the equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)

Tasks, Part 2
Use mileage, cylinders, liters, doors, cruise, sound, and leather to find the linear regression equation.
Find the equation's R2 score (use the .score method) to determine whether the equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)
Find the combination of the factors that is the best predictor for price.


Tasks, Hard Mode
Research dummy variables in scikit-learn to see how to use the make, model, and body type.
Find the best combination of factors to predict price.
[ ]
df = pd.read_csv("car_data.csv")



----------------------------------------------------------------------------------------------------------------------

Assignment5

Ground Cricket Chirps

In The Song of Insects (1948) by George W. Pierce, Pierce mechanically measured the frequency (the number of wing vibrations per second) of chirps (or pulses of sound) made by a striped ground cricket, at various ground temperatures. Since crickets are ectotherms (cold-blooded), the rate of their physiological processes and their overall metabolism are influenced by temperature. Consequently, there is reason to believe that temperature would have a profound effect on aspects of their behavior, such as chirp frequency.



In general, it was found that crickets did not sing at temperatures colder than 60º F. or warmer than 100º F.


[ ]
ground_cricket_data = {"Chirps/Second": [20.0, 16.0, 19.8, 18.4, 17.1, 15.5, 14.7,
                                         15.7, 15.4, 16.3, 15.0, 17.2, 16.0, 17.0,
                                         14.4],
                       "Ground Temperature": [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7,
                                              71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5,
                                              76.3]}
df = pd.DataFrame(ground_cricket_data)


Tasks
Find the linear regression equation for this data.

Chart the original data and the equation on the chart.

Find the equation's R2 score (use the .score method) to determine whether the equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)

Extrapolate data: If the ground temperature reached 95, then at what approximate rate would you expect the crickets to be chirping?
Interpolate data: With a listening device, you discovered that on a particular morning the crickets were chirping at a rate of 18 chirps per second. What was the approximate ground temperature that morning?



Assignment6
Brain vs. Body Weight
In the file brain_body.txt, the average brain and body weight for a number of mammal species are recorded. Load this data into a Pandas data frame.

Tasks
Find the linear regression equation for this data for brain weight to body weight.
Chart the original data and the equation on the chart.
Find the equation's R2 score (use the .score method) to determine whether the equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)


[ ]
df = pd.read_fwf("brain_body.txt")
Assignment7
Salary Discrimination
The file salary.txt contains data for 52 tenure-track professors at a small Midwestern college. This data was used in legal proceedings in the 1980s about discrimination against women in salary.



The data in the file, by column:

Sex. 1 for female, 0 for male.
Rank. 1 for assistant professor, 2 for associate professor, 3 for full professor.
Year. Number of years in current rank.
Degree. Highest degree. 1 for doctorate, 0 for master's.
YSdeg. Years since highest degree was earned.
Salary. Salary/year in dollars.


Tasks
Find the linear regression equation for this data using columns 1-5 to column 6.
Find the selection of columns with the best  R2  score.
Report whether sex is a factor in salary.

---------------------------------------------------------------------------------------------------------------------------
#Assignment 4
 
Boston Housing Dataset
Predicting Median value of owner-occupied homes

The aim of this assignment is to learn the application of machine learning algorithms to data sets. This involves learning what data means, how to handle data, training, cross validation, prediction, testing your model, etc.
This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive, and has been used extensively throughout the literature to benchmark algorithms. The data was originally published by Harrison, D. and Rubinfeld, D.L. Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.
The dataset is small in size with only 506 cases. It can be used to predict the median value of a home, which is done here. There are 14 attributes in each case of the dataset. They are:


CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's

 
Aim

To implement a linear regression with regularization via gradient descent.
to implement gradient descent with Lp norm, for 3 different values of p in (1,2]
To contrast the difference between performance of linear regression Lp norm and L2 norm for these 3 different values.
Tally that the gradient descent for L2 gives same result as matrix inversion based solution.
All the code is written in a single python file. The python program accepts the data directory path as input where the train and test csv files reside. Note that the data directory will contain two files train.csv used to train your model and test.csv for which the output predictions are to be made. The output predictions get written to a file named output.csv. The output.csv file should have two comma separated columns [ID,Output].
Working of Code
NumPy library would be required, so code begins by importing it
Import phi and phi_test from train and test datasets using NumPy's loadtxt function
Import y from train dataset using the loadtxt function
Concatenate coloumn of 1s to right of phi and phi_test
Apply min max scaling on each coloumn of phi and phi_test
Apply log scaling on y
Define a function to calculate change in error function based on phi, w and p norm
Make a dictionary containing filenames as keys and p as values
For each item in this dictionary
Set the w to all 0s
Set an appropriate value for lambda and step size
Calculate new value of w
Repeat steps until error between consecutive ws is less than threshold
Load values of id from test data file
Calculate y for test data using phi test and applying inverse log
Save the ids and y according to filename from dictionary

 
Feature Engineering

Columns of phi are not in same range, this is because their units are different i.e phi is ill conditioned
So, min max scaling for each column is applied to bring them in range 0-1
Same scaling would be required on columns of phi test
Log scaling was used on y. This was determined by trial and error
Comparison of performance
(p1=1.75, p2=1.5, p3=1.3)
As p decreases error in y decreases
As p decreases norm of w increases but this can be taken care by increasing lambda
As p decreases number of iterations required decreases
Tuning of Hyperparameter
If p is fixed and lambda is increased error decreases up to a certain lambda, then it starts rising
So, lambda was tuned by trial and error.
Starting with 0, lambda was increased in small steps until a minimum error was achieved.
Comparison of L2 gradient descent and closed form
Error from L2 Gradient descent were 4.43268 and that from closed form solution was 4.52624.
Errors are comparable so, the L2 gradient descent performs closely with closed form solution.

