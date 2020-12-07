# Shine Bright Like a Diamond

## **Project Aim: To predict diamonds price based on their characteristics (weight, color, quality of cut, etc.), putting into practice machine learning techniques** ##

<img src="https://assets.entrepreneur.com/content/3x2/2000/20160305000536-diamond.jpeg" width="550" height="350">


-> Columns of the dataset include the following features information:
- id: only for test & sample submission files, id for prediction sample identification
- price: price in USD
- carat: weight of the diamond
- cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- color: diamond colour
- clarity: a measurement of how clear the diamond is
- x: length in mm
- y: width in mm
- z: depth in mm
- depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- table: width of top of diamond relative to widest point (43--95)

**Feature Engineering**
- Dropped columns that have high correlation
- Label encoding


**Models**
- Linear Regression
- Random Forest Regressor and Grid Search CV
- KNN Neighbors
- Lasso and Ridge Regressors