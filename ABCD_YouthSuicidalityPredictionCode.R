# youth suicidality prediction code 

# load packages 

library("caret") 
library("caretEnsemble")
library("rpart")
library("ranger")  
library("caTools") 
library("pROC")
library("broom")
library("ggROC")
library("ggsci") 
library("glmnet") 
library("cowplot") # plot_grid

set.seed(7061)

# if parallel 
library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
#stopCluster(cl) 

# load predefined functions

get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

test_roc <- function(model, data) {
  roc_obj <- roc(data$Class, predict(model, data, type = "prob")[, "YES"],
                 levels = c("YES", "NO"))
  ci(roc_obj)
}

suicidal_prediction2 <- function(training, ancestry, outcome, modelnumber, todaysdate){
	modelname <- paste0("models_SuicideControl_", ancestry, "_", outcome, "_model", modelnumber)
	tablename <- paste0("table_SuicideControl_", ancestry, "_", outcome, "_model", modelnumber)
	rocname_glm <- paste0("roc_SuicideControl_", ancestry, "_", outcome, "_GLM_model", modelnumber)
	rocname_rf <- paste0("roc_SuicideControl_", ancestry, "_", outcome, "_RF_model", modelnumber)
	rocname <- paste0("roc_SuicideControl_", ancestry, "_", outcome, "_model", modelnumber)
	savename <- paste0("Results_SuicideControl_", ancestry, "_", outcome, "_model", modelnumber, "_", todaysdate)
	print(paste0(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", modelname,))
	print(paste0(modelname, " is saved!"))
	print(paste0(rocname, " is ready for visualization task!"))
	print(paste0(tablename, " is the final result!")) #c("extratrees", 
	ctrl <- trainControl(method = "repeatedcv", number = 5, repeats=10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = "final", allowParallel = TRUE, verboseIter = FALSE)
	rf.fit <- train(Class~., data = training, method = "ranger", trControl = ctrl, tuneGrid=expand.grid(mtry=c(2,4,7,10,50), splitrule="gini", min.node.size=c(1:2,7)),  metric = "ROC", tuneLength = 20) #importance = "impurity")
	glm.fit <- train(Class ~ ., data = training, trControl = ctrl, method = "glm", family = "binomial", metric = "ROC", na.action=na.pass, preProcess = c("nzv", "center", "scale")) 
	glmnet.fit <- train(Class ~.,data = training, trControl = ctrl, verbose=TRUE, method = "glmnet", metric = "ROC", tuneGrid = expand.grid(alpha = 0:1,lambda = seq(0.0001, 1, length = 20)), preProcess = c("nzv", "center", "scale"))  
	####### performance check! - normal train object 
	temp <- list(GLM=glm.fit, RF=rf.fit, glmnet=glmnet.fit) 
	assign(outcome, list(GLM=glm.fit, RF=rf.fit, glmnet=glmnet.fit), envir=.GlobalEnv)
	temp.preds <- lapply(temp, predict, newdata=test, type="prob")
	temp.preds <- lapply(temp.preds, function(x) x[,"YES"]) 
	temp.preds <- data.frame(temp.preds)
	write.table(temp.preds, paste0(savename, "_RF_Logit_glmnet_values.txt"), quote=FALSE, sep="\t", row.names=FALSE, col.names=TRUE)
	assign(modelname, resamples(temp), envir = .GlobalEnv)
	print(summary(get(modelname)))
	#print(varImp(rf.fit))
	print(varImp(glm.fit)) 
	print(varImp(glmnet.fit))
	print(lapply(temp, confusionMatrix)) 
	temp2 <- lapply(temp, test_roc, data=test)
	temp2 <- lapply(temp2, as.vector)
	temp2 <- do.call("rbind", temp2)
	colnames(temp2) <- c("lower", "ROC", "upper")
	assign(tablename, as.data.frame(temp2), envir=.GlobalEnv)
	print(get(tablename))   
	assign(rocname_glm, roc(test$Class, temp.preds$GLM), envir = .GlobalEnv)  
	assign(rocname, roc(test$Class, temp.preds$glmnet), envir = .GlobalEnv)  
	assign(rocname_rf, roc(test$Class, temp.preds$RF), envir = .GlobalEnv)  
	print(as.vector(ci(get(rocname))))
} 



sink("PredictionResult_EuropeanOnly_SuicideAll_downsampling_SuicideControl_10202021.txt", append=TRUE) 

######################################################### MODEL 4 

data <- merge(cov[,-c(8,9)], prediction_data_eu, by="KEY") #european-ONLY 
data <- merge(data, prs, by="KEY")
data <- merge(data, fes, by="KEY")
data <- data[complete.cases(data$Suicide_all),] # removed NA 
data <- data[complete.cases(data$sex),]  # categorical data (not for imputation) 
data <- data[complete.cases(data$race.ethnicity),] 
data <- data[complete.cases(data$married),]
data <- data[complete.cases(data$fes_q1),]
data <- data[complete.cases(data$fes_q5),] 
imputing <- preProcess(data, method = c("knnImpute"))
dat <- predict(imputing, data) 
# set target class
dat$Class <- ifelse(dat$Suicide_all==TRUE, "YES", "NO") 
dat$Class <- as.factor(dat$Class) 
dat <- dat[,-c(35:39)] # suicide phenotype  
index <- createDataPartition(y = dat$Class, p = .8, list = FALSE)
training <- dat[index, ]
test <- dat[-index, ]
down_train <- downSample(x = training[,-c(ncol(training), ncol(training)-1)], y = training$Class) 

#suicidal_prediction(training, "EA", "overallsuicidality", "4", "10202021")
suicidal_prediction2(down_train, "EA", "overallsuicidality", "4", "10202021")


######################################################### MODEL 3 

data <- merge(cov[,-c(8,9)], prediction_data_eu, by="KEY") #european-ONLY 
data <- merge(data, fes, by="KEY")
data <- data[complete.cases(data$Suicide_all),] # removed NA 
data <- data[complete.cases(data$sex),]  # categorical data (not for imputation) 
data <- data[complete.cases(data$race.ethnicity),] 
data <- data[complete.cases(data$married),]
data <- data[complete.cases(data$fes_q1),]
data <- data[complete.cases(data$fes_q5),] 
imputing <- preProcess(data, method = c("knnImpute"))
dat <- predict(imputing, data) 
# set target class
dat$Class <- ifelse(dat$Suicide_all==TRUE, "YES", "NO") 
dat$Class <- as.factor(dat$Class) 
dat <- dat[,-c(35:39)] # suicide phenotype  
index <- createDataPartition(y = dat$Class, p = .8, list = FALSE)
training <- dat[index, ]
test <- dat[-index, ]
down_train <- downSample(x = training[,-c(ncol(training), ncol(training)-1)], y = training$Class)
#suicidal_prediction(training, "EA", "overallsuicidality", "3", "10202021")
suicidal_prediction2(down_train, "EA", "overallsuicidality", "3", "10202021")

######################################################### MODEL 2  

data <- merge(cov[,-c(8,9)], prediction_data_eu[,c("KEY","Suicide_all")], by="KEY") 
data <- merge(data, prs, by="KEY")
data <- data[complete.cases(data$Suicide_all),] # removed NA 
data <- data[complete.cases(data$sex),]  # categorical data (not for imputation) 
data <- data[complete.cases(data$race.ethnicity),] 
data <- data[complete.cases(data$married),]
imputing <- preProcess(data, method = c("knnImpute"))
dat <- predict(imputing, data) 
# set target class
dat$Class <- ifelse(dat$Suicide_all==TRUE, "YES", "NO") 
dat$Class <- as.factor(dat$Class) 
dat <- dat[,-8] # suicide phenotype  
index <- createDataPartition(y = dat$Class, p = .8, list = FALSE)
training <- dat[index, ]
test <- dat[-index, ]
down_train <- downSample(x = training[,-c(ncol(training), ncol(training)-1)], y = training$Class) # 
#suicidal_prediction(training, "EA", "overallsuicidality", "2", "10202021")
suicidal_prediction2(down_train, "EA", "overallsuicidality", "2", "10202021")

######################################################### MODEL 1

data <- merge(cov[,-c(8,9)], prediction_data_eu[,c("KEY","Suicide_all")], by="KEY") 
data <- data[complete.cases(data$Suicide_all),] # removed NA 
data <- data[complete.cases(data$sex),]  # categorical data (not for imputation) 
data <- data[complete.cases(data$race.ethnicity),] 
data <- data[complete.cases(data$married),]
imputing <- preProcess(data, method = c("knnImpute"))
dat <- predict(imputing, data) 
# set target class
dat$Class <- ifelse(dat$Suicide_all==TRUE, "YES", "NO") 
dat$Class <- as.factor(dat$Class) 
dat <- dat[,-8] # suicide phenotype  
index <- createDataPartition(y = dat$Class, p = .8, list = FALSE)
training <- dat[index, ]
test <- dat[-index, ]
down_train <- downSample(x = training[,-c(ncol(training), ncol(training)-1)], y = training$Class) # 

#suicidal_prediction(training, "EA", "overallsuicidality", "1", "10202021")
suicidal_prediction2(down_train, "EA", "overallsuicidality", "1", "10202021")
sink()

