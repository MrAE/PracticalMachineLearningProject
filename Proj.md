---
title: "Data Science Specialization: Practical Machine Learning"
author: "Jesse Leigh Patsolic"
output: 
  html_document:
    keep_md: true
---


```
## 
## 
## processing file: Proj.Rmd
```

```
## Error in parse_block(g[-1], g[1], params.src): duplicate label 'knitCode'
```

## Objective 

The goal of this project is to use the data provided by 
[groupware](http://groupware.les.inf.puc-rio.br/har) to create a 
model that will predict the type of exercise that each subject is performing.

## Data Preparation 

First, the libraries required to run this code are loaded.
The training and testing data are loaded from the links given on the
project webpage and stored in appropriately named objects.
The first few columns of the data denote subject name and 
various timestamps which are unnecessary for the prediction and 
are removed.


```r
require(caret)
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
require(rattle)
```

```
## Loading required package: rattle
## Loading required package: RGtk2
## Rattle: A free graphical interface for data mining with R.
## Version 3.5.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
require(ggplot2)

if(!file.exists("data")){
    dir.create("data")
    trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(trainURL, destfile="./data/training.csv", method="curl")
    download.file(testURL, destfile="./data/testing.csv", method="curl")
    }

training0 <- read.csv("data/training.csv")[,-c(1:5)]
testing0  <- read.csv("data/testing.csv")[,-c(1:5)]
```

In surveying the data I have noticed there are some entries with
`#DIV/0!` values.  I will convert these to `NA` in an attempt to clean 
the data before modeling.  The columns that have these values have been
read in as factor variables; this is corrected as well.  
Some columns have more `NA` values than actual data entries and are 
removed from the training data.


```r
testing0$new_window <- as.numeric(testing0$new_win)
training1 <- replace(training0, 
                     which(training0 == "#DIV/0!", arr.ind=TRUE),
                     NA)

i <- sapply(training1, is.factor); i[c("classe")] <- !i[c("classe")]
training1[i] <- lapply(training1[i], as.numeric)

l0 <- lapply(training1, function(x) all(!is.na(x)))
nl0 <- names(l0[l0 == TRUE])
training1 <- training1[,nl0]
```
## Training Selection
The cleaned data will be subset further into `train` and `test` sets
for model creation.  


```r
set.seed(648)                          # seed for reproducibility
inTrain <- createDataPartition(y=training1$classe,p=0.05,list=FALSE)
train1 <- training1[inTrain,]
test1 <- training1[-inTrain,]
```

## Fitting the model
In trying different models types it seems that
`randomForest` gave good results, so that is the method I will be using. 
Training the model took a long time using a 70%/30% mix for training and
testing sets respectively.  Even though I train with only 0.05% of the
training data, my results appear to be good.



```r
modFitRF <- train(classe ~ ., method="rf", data=train1, prox=TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


```r
print(modFitRF$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 5.8%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 279   0   0   0   0  0.00000000
## B  14 159  14   3   0  0.16315789
## C   0   7 163   1   1  0.05232558
## D   0   3   6 152   0  0.05590062
## E   0   3   4   1 173  0.04419890
```

The OOB estimate is given as 5.8%, which seems very good.

## Prediction 
Now the prediction will be run on the test set and evaluated.


```r
modPred <- predict(modFitRF, newdata=test1)
table(test1$classe == modPred)
```

```
## 
## FALSE  TRUE 
##  1143 17496
```
The error rate as seen in the table is approximately 6.1%.


```r
confusionMatrix(test1$classe, modPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5279    0    1   20    1
##          B  240 3207  130   25    5
##          C    0  175 3038   31    6
##          D    5   29  247 2754   20
##          E   42   36   28  102 3218
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9387          
##                  95% CI : (0.9351, 0.9421)
##     No Information Rate : 0.2986          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9223          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9484   0.9304   0.8821   0.9393   0.9902
## Specificity            0.9983   0.9737   0.9860   0.9808   0.9865
## Pos Pred Value         0.9958   0.8891   0.9348   0.9015   0.9393
## Neg Pred Value         0.9785   0.9840   0.9736   0.9886   0.9979
## Prevalence             0.2986   0.1849   0.1848   0.1573   0.1744
## Detection Rate         0.2832   0.1721   0.1630   0.1478   0.1726
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9734   0.9520   0.9341   0.9601   0.9883
```

The accuracy is very good, given the comparatively small size of the
training set.  

## Validation
I ran my model on the `testing0` data from above (which is the
validation data).  I report the feedback I got from the submission
script along with my predictions.


```r
Predicted <- predict(modFitRF, newdata=testing0)
truth <- rep(TRUE,20); truth[c(3,11)] <- FALSE
data.frame(Predicted, truth)
```

```
##    Predicted truth
## 1          B  TRUE
## 2          A  TRUE
## 3          A FALSE
## 4          A  TRUE
## 5          A  TRUE
## 6          E  TRUE
## 7          D  TRUE
## 8          B  TRUE
## 9          A  TRUE
## 10         A  TRUE
## 11         B FALSE
## 12         C  TRUE
## 13         B  TRUE
## 14         A  TRUE
## 15         E  TRUE
## 16         E  TRUE
## 17         A  TRUE
## 18         B  TRUE
## 19         B  TRUE
## 20         B  TRUE
```

### Happy Mathing ! ###














