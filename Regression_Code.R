library(caret)
library(randomForest)

## Read in Processed Data
newData <- read.csv("~/Desktop/DATA 599/Data/newData_For_RFE.csv", header = TRUE, row.names = 1)
DFS_time <- read.csv("~/Desktop/DATA 599/Data/DFS_time_data.csv")

probe_sets <- newData[,colnames(newData)[c(1:14923)]]
DFS_time <- DFS_time[, colnames(DFS_time)[c(1)]]
DFS_time <- as.numeric(DFS_time)

## training & test split
set.seed(100)
split <- createDataPartition(DFS_time, p = 0.8, list = FALSE)
adData <- probe_sets
adData$DFS_time <- DFS_time
training <- adData[split, ]
testing <- adData[-split, ]
predVars <- names(probe_sets)

# Set Sequence and Function
varSeq <- c(2:30, 50, 100, 200, 400, 800, 1600, 3000, 5000, 10000, 14923)
newRF <- rfFuncs

# RFE 
set.seed(100)
ctrl <- rfeControl(method = "boot", functions = newRF, rerank = TRUE, number = 50, verbose = TRUE)
start_time <- Sys.time()
rfRFE <- rfe(x = training[, predVars], y = training$DFS_time, sizes = varSeq, rfeControl = ctrl, ntree = 1000)
end_time <- Sys.time()
run_time <- end_time - start_time #report Time difference of 9.389845 hours
save.image("~/Desktop/Experiments/Regression/Untitled.RData")

# Top Variables from RFE
head(rfRFE$optVariables)
plot(rfRFE)
ggplot(rfRFE)
rfRFE  # print out

# 'Selected' model contains all 14,923 variables.  When the "optimal" size by rfe is greater than 30, start with 
# 30 and apply the one-standard-error (or tolerance) rule to the performance metric reported for the size of 30. 

# Update rfe after oneSE # to retrieve subset sizes 2:30, need to use 1:29**
indexSize <- oneSE(rfRFE$results[1:29,], metric = "RMSE", num = 50, maximize = FALSE)
new_rfe <- update(rfRFE, x = training[, predVars], y = training$DFS_time, size = varSeq[indexSize])
plot(new_rfe)

# best variables from updated rfe
opt_rfe_subset <- new_rfe$bestVar

# Random Forests Regression: using the subset() function to only select the opt_rfe_subset 
set.seed(100)
#error in using all predVars.  rfModel1 <- randomForest(subset= opt_rfe_subset, x = training[, predVars], y = training$DFS_time, importance = TRUE, ntree = 1000)
rfModel2 <- randomForest(x = training[, opt_rfe_subset], y = training$DFS_time, importance = TRUE, ntree = 1000)
plot(rfModel2)
varImpPlot(rfModel2, sort=TRUE, n.var=17)
rfPredictOOB <- predict(rfModel2, predict.all = TRUE)
rfValuesOOB <- data.frame(obs = training$DFS_time, pred = rfPredictOOB)
defaultSummary(rfValuesOOB)
rf_OOB_MSE <- mean((rfPredictOOB - rfValuesOOB$obs)^2)
rf_OOB_MSE

# Predicting with training data
rfPredicttrain <- predict(rfModel2, training[, predVars])
rfValuestrain <- data.frame(obs = training$DFS_time, pred = rfPredicttrain)
defaultSummary(rfValuestrain)
MSE <- mean((rfPredicttrain - training$DFS_time)^2)
MSE #print out results

# Predicting with testing data
rfPredictTest <- predict(rfModel2, testing[, predVars])
rfValuesTest <- data.frame(obs = testing$DFS_time, pred = rfPredictTest)
defaultSummary(rfValuesTest)
MSE <- mean((rfPredictTest - testing$DFS_time)^2)
MSE #print out results

# Validate model assumptions with relevant plots

# plot observed vs. predicted values
xyplot(training$DFS_time ~ rfPredicttrain, xlab = "Predicted", ylab = "Observed")

# plot residuals vs. predicted values
residualValues <- training$DFS_time - rfPredicttrain
plot(rfPredicttrain, residualValues, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)

# QQ normality plot for residuals
qqnorm(residualValues, ylab="Standardized Residuals", xlab= "Normal Scores")
qqline(residualValues, col = 2)

## Check normal distribution with K-S test
ktestresults <- ks.test(residualValues, "pnorm", mean=mean(residualValues), sd=sd(residualValues))

##############################################################################
# Given the negative results, performing SVM with opt subset from RF

library(kernlab)
library(caret)
#train model
set.seed(100)
svmRTuned <- train(x = training[, opt_rfe_subset], 
                   y = training$DFS_time, 
                   method = "svmLinear", 
                   tuneLength = 14, 
                   trControl = trainControl(method = "cv"))
svmRTuned$finalModel
#svmRTuned2 <- train(x = training[, opt_rfe_subset], 
#                   y = training$DFS_time, 
#                   method = "svmRadial", 
#                  tuneLength = 14, 
#                  trControl = trainControl(method = "cv"))
plot(varImp(svmRTuned),17) 
varImp(svmRTuned)

# predict with training
svmpredicttrain <- predict(svmRTuned, training[, opt_rfe_subset])
svmValues <- data.frame(obs = training$DFS_time, pred = svmpredicttrain)
defaultSummary(svmValues)
MSE <- mean((svmpredicttrain - training$DFS_time)^2)
MSE #309.8138

#predict with test data
svmpredicttest <- predict(svmRTuned, testing[, opt_rfe_subset])
svmValues1 <- data.frame(obs = testing$DFS_time, pred = svmpredicttest)
defaultSummary(svmValues1)
MSE <- mean((svmpredicttest - testing$DFS_time)^2)
MSE #369.3561
save.image("~/Desktop/Experiments/Regression/Regression_Code.RData")

# validate model assumptions
# plot observed vs. predicted values
xyplot(training$DFS_time ~ svmpredicttrain, xlab = "Predicted", ylab = "Observed")

# plot residuals vs. predicted values
svmresiduals <- training$DFS_time - svmpredicttrain
plot(svmpredicttrain, svmresiduals, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)

# QQ normality plot for residuals
qqnorm(svmresiduals, ylab="Standardized Residuals", xlab= "Normal Scores")
qqline(svmresiduals, col = 2)





