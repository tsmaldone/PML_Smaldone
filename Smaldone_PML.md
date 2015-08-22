# Practical Machine Learning: Prediction Peer Assessment Project
Tony Smaldone  
Friday, August 21, 2015  


### Introduction

Using newest technology it is relatively simple for people to quantify how much of a
particular activity they do; number of steps taken in a day, number of repetitions
of a certain exercise, are but two examples. What is usually not quantified is
not how *well* a particular activity is performed. Doing a particular activity over and over
and not doing it well can provide little benefit and in some cases, can actually 
be detrimental (i.e., can cause injury). This project will use the data from
six study participants who were equipped with accelerometers placed on the belt,
forarm, arm and dumbbell and who were asked to perform barbell lifts correctly
and incorrectly in five different ways to predict the manner in which the participants
did the exercise.

### Data

The data used in this project consists of a *training* and *test* dataset. The
data can be obtained from:

* [Training Data](https://d396qusza40orc.cloudfron.net/predmachlearn/pml-training.csv)
* [Test Data](https://d396qusza40orc.cloudfron.net/predmachlearn/pml-testing.csv)

For more information on the project in which the data was collect refer to 
[Data Source](http://groupware.les.inf.puc-rio.br/har). In particular refer to the section
on the *Weight Lifting Exercise Dataset*. In particular, six participants were asked to 
perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different 
fashions, one exactly per the specification and four with some deviation (i.e., incorrectly).

### Data Analysis Environment

This data analysis aspect of this research project was done using [R](https://www.r-project.org/about.html),
*"a language and environment for statistical computing and graphics"*. All analysis
was done on a Windows-based PC. The specifics of the computing environment upon which this research was conducted is:


```r
sessionInfo()
```

```
## R version 3.0.3 (2014-03-06)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## loaded via a namespace (and not attached):
## [1] digest_0.6.8    evaluate_0.6    formatR_1.1     htmltools_0.2.6
## [5] knitr_1.9       rmarkdown_0.5.1 stringr_0.6.2   tools_3.0.3    
## [9] yaml_2.1.13
```

### R Environment Initialization

Define and set the working directory where the data and any associated resources are located.


```r
setwd("~/Coursera/Practical Machine Learning/Project")
```

Load the R packages, `caret`, `ggplot2` and `AppliedPredictiveModeling`. From [caret](http://topepo.github.io/caret/index.html),
the `caret` package (short for **C**lassification **A**nd **RE**gression **T**raining) is a set of functions that 
attempt to streamline the process for creating predictive models. In addition to other functionality, the package 
contains tools for:

* data splitting
* pre-processing
* feature selection
* model tuning using resampling
* variable importance estimation


```r
library(ggplot2)
library(caret)
```

```
## Loading required package: lattice
```

```r
library(AppliedPredictiveModeling)
```

Read the training and test data sets.


```r
if (!"trainData" %in% ls()) {
  trainData <- read.csv("pml-training.csv")
}
```


```r
if (!"testData" %in% ls()) {
  testData <- read.csv("pml-testing.csv")
}
```


### Data Cleaning / Pre-processing

For the purposes herein, want to be sure that all missing data or data values representing
infinity are removed. In addition, since this project is only interested in the data 
corresponding to the use of dumbbells, only the columns that are relevant to the measurements
of such activity will be included in the data set used to train the model. These columns will
be those which contain belt, (fore)arm, dumbbell.


```r
idBadData <-sapply(trainData,function(x) any(is.na(x) | x == ""))
findPredictors <- !idBadData & grepl("belt|[^(fore)]arm|dumbbell|forearm",names(idBadData))
predictors <-names(idBadData)[findPredictors]
colsToInclude <- c("classe",predictors)
dataToTrainWith <- trainData[,colsToInclude]
```

The predictor will be the `classe` variable. The interpretation of the classe variable is:

* Class A: correct execution per the specification
* Class B: Throwing the elbows to the front
* Class C: Lifting the dumbbell only halfway
* Class D: Lowering the dumbbell only halfway
* Class E: Throwing the hips to the front


### Exploratory Data Analysis And Reproducibility

Exploring the training data set `trainData` 
 

```r
dim(trainData)
```

```
## [1] 19622   160
```

it is observed that it contains 19622 observations and 160 columns. The names of the columns are as defined in the
above reference document. 

The data set `dataToTrainWith` is the processed data set (see above). Exploring `dataToTrainWith`:


```r
dim(dataToTrainWith)
```

```
## [1] 19622    53
```

Note the number of observations are the same, but the number of columns has been greatly reduced.

To see the breakdown of the different classe values:


```r
summary(dataToTrainWith$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

These sum to the 19622 observations in the training data set.

To investigate but one of the predictors, `roll_forearm`, create the following bloxplot and pair-wise plots. To do likewise for all predictors is a bit much for here, so will just use
these as an example.


```r
p1<-qplot(classe,roll_forearm,data=dataToTrainWith,fill=classe,geom=c("boxplot"))
print(p1)
```

![](Smaldone_PML_files/figure-html/unnamed-chunk-10-1.png) 


```r
featurePlot(x=dataToTrainWith[,c("pitch_dumbbell","roll_forearm","roll_arm")],y=dataToTrainWith$classe,plot="pairs")
```

![](Smaldone_PML_files/figure-html/unnamed-chunk-11-1.png) 

To ensure that the results presented herein can be reproduced, will initialize the random
number generator seed:


```r
set.seed(1)
```



### Model Definition / Training

Several different models were created and the results analyzed. For the purposes herein, especially given the imposed 
size limitation of the document, will only reference the model decided to be the training model; i.e., the 
**Random Forest Model**.

Random forests are very good in that it is an ensemble learning method used for classification and regression.  
It uses multiple models for better performance than just using a single tree model. It uses many sample sections
which allows for variable importance to be considered. See [Random Forests](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#features)
for more details.

### Cross Validation

The technique of splitting the training data into two subsets, one to be used to **train** the model
and one used to **test** the model is referred to as the hold-out method. Though convenient,
it has drawbacks, one being that the estimate of the error rate from the prediction may
be misleading should the split of the data result in an unfortunate split.

In lieu of using the hold-out method, K-fold cross validation was used. This technique basically
runs the *experiment* K times and uses K-1 folds for training and the remaining one for
testing. The fold for testing, though of equal size in each experiment, the actual data in
the fold is different from experiment to experiment. But, *a key advantage of K-fold cross
validation is that all the data in the dataset will be used for both training and testing!*

For this project 5-fold cross validation was used and was applied to all the models generated (which as mentioned
above, are not specifically included in this write-up, but were assessed thus leading to the conclusion that Random
Forest was the best approach).


```r
fitControl <- trainControl(method="cv",number=5)
```

### Out Of Sample Error

In simple terms, out of sample error is the error between the predicted values and the actual using a different test set from
which the model was trained. Given that K-fold cross validation was used in the model presented herein, which implies that
all the data will be used, the expectation is that there will be a very low out of sample error. To estimate this would require
averaging the errors for each of the folds. Suffice it here to say that it will be low and will, instead, look at the
*Out-Of-Bag (OOB)* value. Without going into a lot of detail, the OOB error is based on random resamples of the data. It has been shown that 
the OOB 
estimate is as accurate as using a test set of the same size as the training set. A benefit of using the OOB error estimate is that it
removes the need for a set aside test set.

### Model Training

First, will need to *train* the model (the data used to train the model will be per the 5-fold cross validation 
technique):


```r
rfModel <- train(classe ~.,method="rf",data=dataToTrainWith,trControl=fitControl)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

### Model Training Predictions

Use the model to make predictions:


```r
rfModelPrediction <-predict(rfModel,dataToTrainWith)
confusionMatrix(rfModelPrediction,dataToTrainWith$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```


```r
print(rfModel)
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 15697, 15698, 15697, 15698, 15698 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9944451  0.9929733  0.0013879679  0.0017560726
##   27    0.9940373  0.9924572  0.0014026311  0.0017748549
##   52    0.9883804  0.9853006  0.0005866686  0.0007426473
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```


```r
print(rfModel$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.39%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5578    2    0    0    0 0.0003584229
## B   11 3784    2    0    0 0.0034237556
## C    0   14 3406    2    0 0.0046756283
## D    0    0   39 3174    3 0.0130597015
## E    0    0    0    4 3603 0.0011089548
```

To visually see the model performance, plot the model:


```r
plot(rfModel,main="Forest Tree Final Model",xlab="Number of Randomly Selected Predictors")
```

![](Smaldone_PML_files/figure-html/unnamed-chunk-18-1.png) 

The plot clearly shows the high level of accuracy of the Random Forest model using cross validation.

In addition, the following plots error versus the number of trees used in the model:


```r
plot(rfModel$finalModel,log="y",main="Errors Vs. Trees")
```

![](Smaldone_PML_files/figure-html/unnamed-chunk-19-1.png) 

As the plot indicates, the number of errors in the model is extremely small and gets continuously smaller as the
number of trees increases.

Of particular note is the Out Of Bag error (OOB) of roughly .4%. This is extremely low (as predicted it would be above), thus further
evidence of an accurate model.


### Model Testing

Use the developed model to predict the outcome of the given 20 test cases:


```r
hat<-predict(rfModel,testData)
```


```r
hat
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Appendix

To show a comparison to the Random Forest model selected above, here is a simple linear discriminant analysis (lda) model.
First split the training data into a training subset and a testing subset (this was not necessary when using the
K-fold cross-validation technique above).


```r
inTrain<-createDataPartition(dataToTrainWith$classe,p=3/4)[[1]]
training<-dataToTrainWith[inTrain,]
testing<-dataToTrainWith[-inTrain,]
```

Display the proportion of data in the training and testing datasets:



```r
dim(training)
```

```
## [1] 14718    53
```

```r
dim(testing)
```

```
## [1] 4904   53
```



```r
ldaModel <- train(classe ~.,method="lda",data=training)
```

```
## Loading required package: MASS
```

```r
ldaModelPrediction<-predict(ldaModel,training)
confusionMatrix(ldaModelPrediction,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3423  434  267  144  109
##          B   91 1829  236  104  466
##          C  338  336 1697  273  255
##          D  321  113  303 1792  241
##          E   12  136   64   99 1635
## 
## Overall Statistics
##                                           
##                Accuracy : 0.705           
##                  95% CI : (0.6975, 0.7123)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6266          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8179   0.6422   0.6611   0.7430   0.6042
## Specificity            0.9094   0.9244   0.9011   0.9205   0.9741
## Pos Pred Value         0.7820   0.6709   0.5854   0.6469   0.8402
## Neg Pred Value         0.9263   0.9150   0.9264   0.9481   0.9161
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2326   0.1243   0.1153   0.1218   0.1111
## Detection Prevalence   0.2974   0.1852   0.1970   0.1882   0.1322
## Balanced Accuracy      0.8637   0.7833   0.7811   0.8317   0.7892
```

As the results show, the accuracy was only roughly 70%. The predictions against the supplied test data are:


```r
hat1<-predict(ldaModel,testData)
hat1
```

```
##  [1] B A B C C C D D A A D A B A E A A B B B
## Levels: A B C D E
```

Comparing this to the predicted values from the Random Forest model, one can see that there are significant differences.







