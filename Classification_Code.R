## Classification Experiments

#load libraries
library(caret)
library(randomForest)
library(kernlab)

## Read in Processed Data
newData <- read.csv("~/Desktop/Experiments/Classification/Filtered_for_RFE.csv")

## transpose & split variables & DFS_time
data <- as.data.frame(t(newData))
names(data) <- data[1,]
data <- data[-1,]
data <- data[,-1]
Data_For_RFE <- write.csv(data, file = "Data_For_RFE.csv")
###############################################################################
#read data in for RFE experiments
Data_For_RFE <- read.csv("~/Desktop/Experiments/Classification/Data_For_RFE.csv", header = TRUE, row.names = 1)
rfsEvent <- read.csv("~/Desktop/Experiments/Classification/rfsEvent2.csv")

#Variable & Class Labels
probe_sets <- Data_For_RFE[,colnames(Data_For_RFE)[c(1:21852)]]
rfsEvent <- as.factor(rfsEvent[1:397,1])

## training & test split
set.seed(100)
split <- createDataPartition(rfsEvent, p = 0.8, list = FALSE)
adData <- probe_sets
adData$Class <- rfsEvent
adData$Class <- as.factor(adData$Class)
training <- adData[split, ]
testing <- adData[-split, ]
predVars <- names(probe_sets)

## RF-RFE Controls
fiveStats <- function(...) c(twoClassSummary(...),defaultSummary(...))
varSeq <- c(2:30, 50, 75, 100, 200, 400, 800, 1600, 3000, 5000, 10000, 15000, 17500, 21852)
newRF <- rfFuncs
newRF$summary <- fiveStats
ctrl <- rfeControl(method = "boot",
                   functions = newRF,
                   rerank = TRUE,
                   number = 50,
                   verbose = TRUE)
## RF-RFE 
set.seed(100)
start_time <- Sys.time()
rfRFE <- rfe(x = training[, predVars],
             y = training$Class,
             sizes = varSeq,
             metric = "ROC",
             rfeControl = ctrl,
             ntree = 1000)
end_time <- Sys.time()
run_time <- end_time - start_time #report Time difference of 7.492043 hours

# Top Variables from RFE
head(rfRFE$optVariables)
ggplot(rfRFE)
ggplot(rfRFE)
rfRFE # print out

# Discuss opt. results and need for simpler model
# Update rfe after oneSE
indexSize <- oneSE(rfRFE$results[1:29,], metric = "Accuracy", num = 50, maximize = TRUE)
new_rfe <- update(rfRFE, x = training[, predVars], y = training$Class, size = varSeq[indexSize])

# best variables from updated rfe
opt_rfe_subset <- new_rfe$bestVar
ggplot(new_rfe)

# Random Forests Classification Experiments
set.seed(100)
# need to put subset in training rfModel1 <- randomForest(subset= opt_rfe_subset, x = training[, predVars], y = training$Class, importance = TRUE, ntree = 1000)
rfModel2 <- randomForest(x = training[, opt_rfe_subset], y = training$Class, importance = TRUE, ntree = 1000)

varImpPlot(rfModel2, sort=TRUE, n.var=19)

#model Accuracy
rfPredictOOB <- predict(rfModel2, predict.all = TRUE)
rfValuesOOB <- data.frame(obs = training$Class, pred = rfPredictOOB)
defaultSummary(rfValuesOOB)
confusionMatrix(rfPredictOOB,training$Class, positive = "1")

# Predicting with training data
# rfPredicttrain <- predict(rfModel1, training[, predVars])
# rfValuestrain <- data.frame(obs = training$Class, pred = rfPredicttrain)
# defaultSummary(rfValuestrain)
# rfValuestrain   # Can discuss
# confusionMatrix(rfPredicttrain,training$Class, positive = "1")

# Predicting with testing data
rfPredictTest <- predict(rfModel2, testing[, predVars])
rfValuesTest <- data.frame(obs = testing$Class, pred = rfPredictTest)
defaultSummary(rfValuesTest)
rfValuesTest
confusionMatrix(rfPredictTest,testing$Class, positive = "1")

###############################################################################
# SVM Classification with rf-RFE opt variables given multiple failed SVM-RFE attempts
library(kernlab)
library(caret)
#train model
set.seed(100)
svmRTuned <- train(x = training[, opt_rfe_subset], 
                   y = training$Class, 
                   method = "svmLinear", 
                   tuneLength = 14, 
                   trControl = trainControl(method = "boot",
                                            number = 500))
svmRTuned$finalModel
plot(varImp(svmRTuned),19) 
varImp(svmRTuned)

# predict with training
svmpredicttrain <- predict(svmRTuned, training[, opt_rfe_subset])
svmValues <- data.frame(obs = training$Class, pred = svmpredicttrain)
defaultSummary(svmValues)
confusionMatrix(svmpredicttrain,training$Class, positive = "1")

#predict with test data
svmpredicttest <- predict(svmRTuned, testing[, opt_rfe_subset])
svmValues1 <- data.frame(obs = testing$Class, pred = svmpredicttest)
defaultSummary(svmValues1)
confusionMatrix(svmpredicttest,testing$Class, positive = "1")

save.image("~/Desktop/Experiments/Classification/rfRFE_results.RData")
