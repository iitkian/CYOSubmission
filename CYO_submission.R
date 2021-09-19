## ----setup, include=FALSE---------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(fig.width=6, fig.height=4.5)


## ----basic options----------------------------------------------------------------------------------------------------------
### Basic options: clear list, show 4 decimal places
rm(list=ls())
options(digits=4)


## ----installing requisite packages, message=FALSE, warning=TRUE-------------------------------------------------------------
### Installing required packages
if(!require(tidyverse)) install.packages("tidyverse", 
    repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
    repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
    repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", 
    repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", 
    repos = "http://cran.us.r-project.org")
if(!require(HDclassif)) install.packages("HDclassif", 
    repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", 
    repos = "http://cran.us.r-project.org")    
if(!require(C50)) install.packages("C50", 
    repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", 
    repos = "http://cran.us.r-project.org")
if(!require(ipred)) install.packages("ipred", 
    repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", 
    repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", 
    repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", 
    repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", 
    repos = "http://cran.us.r-project.org")


## ----loading requisite libraries--------------------------------------------------------------------------------------------
### Loading required libraries
library(plyr)
library(tidyverse)
library(caret)
library(data.table)
library(rpart)
library(rpart.plot)
library(HDclassif)
library(naivebayes)
library(C50)
library(e1071)
library(ipred)
library(gam)
library(kernlab)
library(randomForest)


## ----dataset download-------------------------------------------------------------------------------------------------------
### Downloading the dataset
link <- "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
db = read.csv(link)


## ----data cleaning----------------------------------------------------------------------------------------------------------
### Inspecting the dataset
head(db) # Look at the first 6 rows
dim(db) # Look at the dataset dimensions


## ----adding column names----------------------------------------------------------------------------------------------------
### Adding column names to the dataset
colnames(db)=c("buying","maint","doors","persons","lug_boot",
    "safety","class")


## ----removing rows with NA values-------------------------------------------------------------------------------------------
### Cleaning data by removing rows with NA values
clean <- db[complete.cases(db),]


## ----inspecting cleaned DS--------------------------------------------------------------------------------------------------
### Inspecting the cleaned dataset
clean %>% head() # Look at the first 6 rows
dim(clean) # Look at the cleaned dataset dimensions


## ----rearranging dataset values for improved graph visualisation------------------------------------------------------------
### Rearrange dataset values to improve graph visualisation
clean$buying <- factor(clean$buying,levels = c("low", "med", 
                "high", "vhigh"))
clean$maint <- factor(clean$maint,levels = c("low", "med", 
                "high", "vhigh"))
clean$class <- factor(clean$class,levels = c("unacc", "acc", 
                "good", "vgood"))
clean$safety <- factor(clean$safety,levels = c("low", "med", 
                "high"))
clean$lug_boot <- factor(clean$lug_boot,levels = c("small", 
                "med", "big"))


### Data exploration and visualisation
### Class vs. count, filled with attributes
clean %>% ggplot(aes(x=class))+geom_histogram(stat="count")+
            theme_classic()


## ----plotting histograms -------------------------------------------------------------------------------------
clean %>% ggplot(aes(x=class,fill=buying))+
            geom_histogram(stat="count")+
            theme_classic()


clean %>% ggplot(aes(x=class,fill=maint))+
            geom_histogram(stat="count")+
            theme_classic()


clean %>% ggplot(aes(x=class,fill=doors))+
            geom_histogram(stat="count")+
            theme_classic()


clean %>% ggplot(aes(x=class,fill=persons))+
            geom_histogram(stat="count")+
            theme_classic()


clean %>% ggplot(aes(x=class,fill=lug_boot))+
            geom_histogram(stat="count")+
            theme_classic()


clean %>% ggplot(aes(x=class,fill=safety))+
            geom_histogram(stat="count")+
            theme_classic()


### Count vs. attribute, filled with class
clean %>% ggplot(aes(x=buying,fill=class))+
            geom_histogram(stat="count")+
            coord_flip()+theme_classic()


clean %>% ggplot(aes(x=maint,fill=class))+
            geom_histogram(stat="count")+
            coord_flip()+theme_classic()


clean %>% ggplot(aes(x=doors,fill=class))+
            geom_histogram(stat="count")+
            coord_flip()+theme_classic()


clean %>% ggplot(aes(x=persons,fill=class))+
            geom_histogram(stat="count")+
            coord_flip()+theme_classic()


clean %>% ggplot(aes(x=lug_boot,fill=class))+
            geom_histogram(stat="count")+
            coord_flip()+theme_classic()


clean %>% ggplot(aes(x=safety,fill=class))+
            geom_histogram(stat="count")+
            coord_flip()+theme_classic()


## ----dividing DS into working and validation--------------------------------------------------------------------------------
### Dividing the dataset into working and validation datasets. 
# Validation dataset will not be used till the end.
set.seed(1)
test_index <- createDataPartition(y = clean$buying, 
                times = 1, p = 0.1, list = FALSE)
working <- clean[-test_index,]
validation <- clean[test_index,]


## ----checking dimensions of working and validation--------------------------------------------------------------------------
### Check dimensions of working and validation datasets to 
# ensure that the partitioning has been done as intended.
dim(working)
dim(validation)


## ----dividing working into train and test-----------------------------------------------------------------------------------
### Dividing the working dataset into train and test datasets. 
# The train dataset is used to train the algorithms, and the 
# test dataset is used to test the algorithms and choose the 
# one with the highest accuracy. The selected algorithm(s)
# will be validated on the validation dataset at the end.
set.seed(1)
test_index <- createDataPartition(y = working$buying, 
                times = 1, p = 0.1, list = FALSE)
train <- working[-test_index,]
test <- working[test_index,]


## ----checking dimensions of train and test----------------------------------------------------------------------------------
### Check dimensions of train and test datasets to 
# ensure that the partitioning has been done as 
# intended.
dim(train)
dim(test)


## ----making decision tree ------------------------------------------------------------------------------------
################## Making decision tree - 1 ##################
set.seed(1)
fit<-rpart(formula = class ~ .,data=train,method = "class")
rpart.plot(fit)


## ----making decision tree 2 ----------------------------------------------------------------------------------
################## Making decision tree - 2 ##################
set.seed(1)
dtree_fit <- train(class ~., data = train, method = "rpart",
                   parms = list(split = "information"),
                   tuneLength = 10)
dtree_fit
prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)


## ----inspecting confusion matrix -----------------------------------------------------------------------------
# Inspecting the confusion matrix
pred <- predict(dtree_fit, newdata = test)
confusionMatrix(pred, as.factor(test$class))


## ----train knn -----------------------------------------------------------------------------------------------
### Training and testing various algorithms ###
################### knn ###################
set.seed(1)
fit <- train(class ~ ., method = "knn", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- data_frame(method = "knn", accuracy = acc, 
                        F1unacc = F1unacc, F1acc = F1acc, 
                        F1good = F1good, F1vgood = F1vgood)

accuracy %>% knitr::kable()


## ----train lda -----------------------------------------------------------------------------------------------
################### lda ###################
set.seed(1)
fit <- train(class ~ ., method = "lda", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="lda",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train naive_bayes ---------------------------------------------------------------------------------------
################### naive_bayes ###################
set.seed(1)
fit <- train(class ~ ., method = "naive_bayes", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="naive_bayes",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train svmLinear -----------------------------------------------------------------------------------------
################### svmLinear ###################
set.seed(1)
fit <- train(class ~ ., method = "svmLinear", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="svmLinear",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train treebag -------------------------------------------------------------------------------------------
################### treebag ###################
set.seed(1)
fit <- train(class ~ ., method = "treebag", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="treebag",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train gamLoess ------------------------------------------------------------------------------------------
################### gamLoess ###################
set.seed(1)
fit <- train(class ~ ., method = "gamLoess", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="gamLoess",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train gam -----------------------------------------------------------------------------------------------
################### gam ###################
set.seed(1)
fit <- train(class ~ ., method = "gam", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="gam",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train svmRadial -----------------------------------------------------------------------------------------
################### svmRadial ###################
set.seed(1)
fit <- train(class ~ ., method = "svmRadial", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="svmRadial",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train hdda ----------------------------------------------------------------------------------------------
################### hdda ###################
set.seed(1)
fit <- train(class ~ ., method = "hdda", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="hdda",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train C5.0Tree ------------------------------------------------------------------------------------------
################### C5.0Tree ###################
set.seed(1)
fit <- train(class ~ ., method = "C5.0Tree", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="C5.0Tree",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----train rf ------------------------------------------------------------------------------------------------
################### rf ###################
set.seed(1)
fit <- train(class ~ ., method = "rf", data = train)
pred <- predict(fit, newdata = test)
cM <- confusionMatrix(pred, as.factor(test$class))
cM
acc <- cM$overall['Accuracy'] 
F1unacc <- cM$byClass[1,'F1']
F1acc <- cM$byClass[2,'F1']
F1good <- cM$byClass[3,'F1']
F1vgood <- cM$byClass[4,'F1']

accuracy <- bind_rows(accuracy,
                      data_frame(method="rf",
                            accuracy = acc, 
                            F1unacc = F1unacc, F1acc = F1acc, 
                            F1good = F1good, F1vgood = F1vgood))
accuracy %>% knitr::kable()


## ----Validation ----------------------------------------------------------------------------------------------
################### Validation ###################
# We train the svmRadial method on the 
# complete working dataset to improve its accuracy. 
# The trained algorithm is then used to predict the 
# car classes in the validation dataset by considering the 
# vehicle characteristics. This is then compared to the 
# actual values in the validation dataset to compute
# the accuracy and the confusion matrix of the algorithm.
################### svmRadial ###################
set.seed(1)
fit <- train(class ~ ., method = "svmRadial", data = working)
pred <- predict(fit, newdata = validation)
confusionMatrix(pred, as.factor(validation$class))

################### End of validation ###################

