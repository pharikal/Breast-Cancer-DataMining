#Close plots
dev.off()
graphics.off()
#Wipe environment
rm(list=ls()) 

#Loading Libraries
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org") 
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org") 
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org") 
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org") 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org") 
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org") 
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org") 
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org") 
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org") 
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org") 
if(!require(Momocs)) install.packages("Momocs", repos = "http://cran.us.r-project.org") 
library(funModeling) 
library(corrplot)
library(factoextra)
library(reshape2)

#Read the dataset
url<- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names <- c('id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
           'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
           'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
           'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst')
cancer_data <- read.csv(url, sep = ",", col.names = names, header = FALSE, 
                        stringsAsFactors=FALSE) %>% as.data.frame

#Data Understading

str(cancer_data)
head(cancer_data)

cancer_data$diagnosis <- as.factor(cancer_data$diagnosis)


#Id value has no impact on class predication, so removing this
cancer_data[,1] <- NULL

#By observing our dataset, it was found that it contains 569 observations with 32 variables.
dim(cancer_data)

#Check if the dataset has any missing value
map(cancer_data, function(.x) sum(is.na(.x)))

#Check data proportions 
diagnosis.table <- table(cancer_data$diagnosis)
prop.table(diagnosis.table)

#Plot the data distribution using a pie chart
colors <- terrain.colors(2)
diagnosis.prop.table <- prop.table(diagnosis.table)*100
diagnosis.prop.df <- as.data.frame(diagnosis.prop.table)
pielabels <- sprintf("%s - %3.1f%s", diagnosis.prop.df[,1], diagnosis.prop.table, "%")
pie(diagnosis.prop.table,
    labels=pielabels,  
    clockwise=TRUE,
    col=colors,
    border="gainsboro",
    radius=0.8,
    cex=0.8, 
    main="Frequency of Cancer Diagnosis")
legend(1, .4, legend=diagnosis.prop.df[,1], cex = 0.7, fill = colors)

#Visualise distribution of data via histograms
#Break up columns into groups, according to their suffix designation (_mean, _se,and __worst) 
#to perform visualisation plots off.
data_mean <- cancer_data[ ,c("diagnosis", "radius_mean", "texture_mean","perimeter_mean", 
                                "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", 
                                "concave_points_mean", "symmetry_mean", "fractal_dimension_mean" )]

data_se <- cancer_data[ ,c("diagnosis", "radius_se", "texture_se","perimeter_se", "area_se", 
                              "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
                              "symmetry_se", "fractal_dimension_se" )]

data_worst <- cancer_data[ ,c("diagnosis", "radius_worst", "texture_worst","perimeter_worst",
                                 "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                                 "concave_points_worst", "symmetry_worst", "fractal_dimension_worst" )]

#Plot histograms of "_mean" variables group by diagnosis
ggplot(data = melt(data_mean, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales ='free_x')

#Plot histograms of "_se" variables group by diagnosis
ggplot(data = melt(data_se, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x')

#Plot histograms of "_worst" variables group by diagnosis
ggplot(data = melt(data_worst, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x')

#Check if the is any correlation between variables as machine learning algorithms 
#assume that the predictor variables are independent from each others.

# Correlation Matrix 
correlationMatrix <- cor(cancer_data[,2:ncol(cancer_data)]) 
corrplot(correlationMatrix, order = "hclust", tl.cex = 0.5, addrect = 8)

# Attributes that are highly corrected (ideally >0.90)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9) 
# Indices of highly correlated attributes 
print(highlyCorrelated)

# Remove correlated variables 
cancer_data_wcor <- cancer_data[,2:ncol(cancer_data)] %>%select(-highlyCorrelated) 
# Check column count after removing correlated variables 
ncol(cancer_data_wcor)

diagnosis <- cancer_data$diagnosis

#Data Modelling
# Plot of PCA 
pca_res_data1 <- prcomp(cancer_data[,2:ncol(cancer_data)], center = TRUE, scale = TRUE) 
plot(pca_res_data1, type="l", main = "")
grid(nx = 10, ny = 14)
title(main = "Principal Component Analysis 1", sub = NULL, xlab = "Components")
box()
summary(pca_res_data1)

#Calculate the proportion of variance
pca_var <- pca_res_data1$sdev^2
pve_df <- pca_var / sum(pca_var)
cum_pve <- cumsum(pve_df)
pve_table <- tibble(comp = seq(1:ncol(cancer_data %>% select(-diagnosis))), pve_df, cum_pve)
ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0)

fviz_pca_biplot(pca_res_data1, col.ind = diagnosis, col="black",
                palette = "jco", geom = "point", repel=TRUE,
                legend.title="Diagnosis", addEllipses = TRUE)

# Plot of PCA
pca_res_data2 <- prcomp(cancer_data_wcor, center = TRUE, scale = TRUE) 
plot(pca_res_data2, type="l", main = "")
grid(nx = 10, ny = 14)
title(main = "Principal Component Analysis 2", sub = NULL, xlab = "Components")
box()
summary(pca_res_data2)

#Calculate the proportion of variance
pca_var <- pca_res_data2$sdev^2
pve_df <- pca_var / sum(pca_var)
cum_pve <- cumsum(pve_df)
pve_table <- tibble(comp = seq(1:ncol(cancer_data_wcor)), pve_df, cum_pve)
ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0)


fviz_pca_biplot(pca_res_data2, col.ind = diagnosis, col="black",
                palette = "jco", geom = "point", repel=TRUE,
                legend.title="Diagnosis", addEllipses = TRUE)

#Compare plot of Diagnosis between PC1 and PC2
pca_df <- as.data.frame(pca_res_data2$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col= diagnosis)) + geom_point(alpha=0.5)

g_pc1 <- ggplot(pca_df, aes(x=PC1, fill= diagnosis)) +
  geom_density(alpha=0.25) 
g_pc2 <- ggplot(pca_df, aes(x=PC2, fill= diagnosis)) +
  geom_density(alpha=0.25)
grid.arrange(g_pc1, g_pc2, ncol=2)

#Let's visuzalize the first 3 components.
df_pcs <- cbind(as_tibble(diagnosis), as_tibble(pca_res_data2$x))
GGally::ggpairs(df_pcs, columns = 2:4, ggplot2::aes(color = value))


#Linear Discriminant Analysis (LDA) 
lda_res_data <- MASS::lda(diagnosis~., data = cancer_data, center = TRUE, scale = TRUE) 
lda_res_data

# Data frame of the LDA for visualization purposes 
lda_df_predict <- predict(lda_res_data, cancer_data)$x %>% as.data.frame() %>% cbind(diagnosis= diagnosis)

# LDA Plot 
ggplot(lda_df_predict, aes(x=LD1, fill=diagnosis)) + geom_density(alpha=0.5) + 
  xlab("LD1") + ylab("Density") + labs(fill = "Diagnosis")

#Data Preparation
# Preparing Train and Test datasets 
set.seed(1815)

model_data <- cbind (diagnosis=diagnosis, cancer_data_wcor)
model_sampling_index <- createDataPartition(model_data$diagnosis, p=0.8, list = FALSE) 
train_data <- model_data[model_sampling_index, ] 
test_data <- model_data[-model_sampling_index, ]

dim(train_data)
dim(test_data)

require(caret)

# Control the computational nuances, number of folds or 
# number of resampling iterations of the train function

fitControl <- trainControl(method="cv", number = 15, classProbs = TRUE, summaryFunction = twoClassSummary)

#Naive Bayes Model
# Confusion Matrix 
model_naiveb <- train(diagnosis~., 
                      train_data,
                      method="nb", 
                      metric="ROC",
                      preProcess=c('center', 'scale'), 
                      #in order to normalize the data 
                      trace=FALSE,
                      trControl=fitControl)

# Check Results
prediction_naiveb <- predict(model_naiveb, test_data)
confusionmatrix_naiveb <- confusionMatrix(prediction_naiveb, test_data$diagnosis, positive = "M") 
confusionmatrix_naiveb

# Plot of Naive Bayes
plot(varImp(model_naiveb), top=10, main="Top Variables- Naive Bayes")


#Logistic Regression Model
# Confusion Matrix 
model_logreg<- train(diagnosis ~., data = train_data, method = "glm",
                     metric = "ROC", 
                     preProcess = c("scale", "center"), 
                     # in order to normalize the data 
                     trControl= fitControl) 
prediction_logreg<- predict(model_logreg, test_data)

# Check Results
confusionmatrix_logreg <- confusionMatrix(prediction_logreg, test_data$diagnosis, positive = "M") 
confusionmatrix_logreg

# Plot of Log Regression 
plot(varImp(model_logreg), top=10, main="Top Variables - Log Regression")

#Logistic Regression Model PCA
# Confusion Matrix 
model_logreg_pca<- train(diagnosis ~., data = train_data, method = "glm",
                     metric = "ROC", 
                     preProcess = c("scale", "center", "pca"), 
                     # in order to normalize the data 
                     trControl= fitControl) 
prediction_logreg_pca<- predict(model_logreg_pca, test_data)

# Check Results
confusionmatrix_logreg_pca <- confusionMatrix(prediction_logreg_pca, test_data$diagnosis, positive = "M") 
confusionmatrix_logreg_pca

# Plot of Log Regression with PCA
plot(varImp(model_logreg_pca), top=10, main="Top Lariables - Log Regression with PCA")

#SVM with radial kernel
model_svm <- train(diagnosis~.,
                   train_data,
                   method="svmRadial",
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trControl=fitControl)
prediction_svm <- predict(model_svm, test_data)

# Check Results 
confusionmatrix_svm <- confusionMatrix(prediction_svm, test_data$diagnosis, positive = "M") 
confusionmatrix_svm

# Plot of SVM
plot(varImp(model_svm), top=10, main="Top Variables- SVM")


#Random Forest Model
# Confusion Matrix
model_randomforest <- train(diagnosis~.,
                            train_data, 
                            method="rf", #also recommended ranger, 
                            # because it is a lot faster than original randomForest (rf) 
                            metric="ROC", 
                            #tuneLength=10, 
                            #tuneGrid = expand.grid(mtry = c(2, 3, 6)), 
                            preProcess = c('center', 'scale'), 
                            trControl=fitControl)
prediction_randomforest <- predict(model_randomforest, test_data)

# Check Results 
confusionmatrix_randomforest <- confusionMatrix(prediction_randomforest, test_data$diagnosis,
                                                positive = "M") 
confusionmatrix_randomforest

# Plot of Random Forest 
plot(varImp(model_randomforest), top=10, main="Top Variables- Random Forest")

#Random Forest Model with PCA
# Confusion Matrix
model_randomforest_pca <- train(diagnosis~.,
                            train_data, 
                            method="rf", #also recommended ranger, 
                            # because it is a lot faster than original randomForest (rf) 
                            metric="ROC", 
                            #tuneLength=10, 
                            #tuneGrid = expand.grid(mtry = c(2, 3, 6)), 
                            preProcess = c('center', 'scale', 'pca'), 
                            trControl=fitControl)
prediction_randomforest_pca <- predict(model_randomforest_pca, test_data)

# Check Results 
confusionmatrix_randomforest_pca <- confusionMatrix(prediction_randomforest_pca,
                                                    test_data$diagnosis, positive = "M") 
confusionmatrix_randomforest_pca

# Plot of Random Forest with PCA
plot(varImp(model_randomforest_pca), top=10, main="Top Variables- Random Forest with PCA")

#KNN
# Confusion Matrix 
model_knn <- train(diagnosis~., 
                   train_data, method="knn", 
                   metric="ROC", 
                   preProcess = c('center', 'scale'), 
                   tuneLength=10, 
                   #The tuneLength parameter #tells the algorithm to try different default v 
                   #alues for the main parameter, in this case 10 default values are used 
                   trControl=fitControl)
# Check Results 
prediction_knn <- predict(model_knn, test_data) 
confusionmatrix_knn <- confusionMatrix(prediction_knn, test_data$diagnosis, positive = "M") 
confusionmatrix_knn

# Plot of KNN 
plot(varImp(model_knn), top=10, main="Top Variables - KNN")

# Neural Network
# Confusion Matrix 
model_nnet <- train(diagnosis~.,
                        train_data,
                        method="nnet", 
                        metric="ROC", 
                        preProcess=c('center', 'scale'), 
                        tuneLength=10, 
                        trace=FALSE, 
                        trControl=fitControl)
# Check Results 
prediction_nnet <- predict(model_nnet, test_data)
confusionmatrix_nnet <- confusionMatrix(prediction_nnet, test_data$diagnosis,
                                            positive = "M") 
confusionmatrix_nnet

# Plot of NNET 
plot(varImp(model_nnet), top= 10, main="Top Variables - NNET")

# Neural Network with PCA
# Confusion Matrix 
model_nnet_pca <- train(diagnosis~.,
                        train_data,
                        method="nnet", 
                        metric="ROC", 
                        preProcess=c('center', 'scale', 'pca'), 
                        tuneLength=10, 
                        trace=FALSE, 
                        trControl=fitControl)
# Check Results 
prediction_nnet_pca <- predict(model_nnet_pca, test_data)
confusionmatrix_nnet_pca <- confusionMatrix(prediction_nnet_pca, test_data$diagnosis,
                                             positive = "M") 
confusionmatrix_nnet_pca

# Plot of NNET PCA 
plot(varImp(model_nnet_pca), top= 10, main="Top Variables - NNET with PCA")

#Neural Network with LDA Model
# Preparing Train and Test datasets 
train_data_lda <- lda_df_predict[model_sampling_index, ] 
test_data_lda <- lda_df_predict[-model_sampling_index, ]

# Confusion Matrix 
model_nnet_lda <- train(diagnosis~.,
                        train_data_lda,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
# Check Results 
prediction_nnet_lda <- predict(model_nnet_lda, test_data_lda) 
confusionmatrix_nnet_lda <- confusionMatrix( prediction_nnet_lda, test_data_lda$diagnosis,
                                             positive = "M")
confusionmatrix_nnet_lda


#Results
# Gather all model results 
models_list <- list(Naive_Bayes= model_naiveb,
                    Logistic_regr= model_logreg,
                    Logistic_regr_PCA= model_logreg_pca,
                    SVM = model_svm,
                    Random_Forest= model_randomforest,
                    Random_Forest_PCA= model_randomforest_pca,
                    KNN= model_knn,
                    Neural = model_nnet,
                    Neural_PCA= model_nnet_pca,
                    Neural_LDA= model_nnet_lda)
models_results <- resamples(models_list)
summary(models_results)

# Plot of Results 
bwplot(models_results, metric="ROC")

# Confusion Matrix results of all models
confusionmatrix_list <- list(
  Naive_Bayes = confusionmatrix_naiveb,
  Logistic_regr = confusionmatrix_logreg,
  Logistic_regr_PCA = confusionmatrix_logreg_pca,
  SVM = confusionmatrix_svm,
  Random_Forest = confusionmatrix_randomforest,
  Random_Forest_PCA = confusionmatrix_randomforest_pca,
  KNN = confusionmatrix_knn,
  Neural = confusionmatrix_nnet,
  Neural_PCA = confusionmatrix_nnet_pca,
  Neural_LDA = confusionmatrix_nnet_lda) 
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results %>% knitr::kable()


#Further analysis with more layers of Neurons
graphics.off()

#Loading required library
library(keras)
library(tensorflow)

### Generates random indices for training set
index <- sample(1:nrow(model_data), nrow(train_data))

### Creates and transforms training set
train_matix<-data.matrix(train_data)
train.x <- train_matix[, 2:21 ] %>%
  as.numeric %>%
  array(dim = c(nrow(train_data), 20))


train.y <- train_matix[, 1]
train.y[train.y[] == "B"] <- 0
train.y[train.y[] == "M"] <- 1

train.y <- to_categorical(train.y) %>%as.array()
train.y <- train.y[, 2]

### Creates and transforms test set
test_matix<-data.matrix(test_data)
test.x <- test_matix[,2:21] %>%
  as.numeric %>%
  array(dim = c(nrow(test_data), 20))

test.y <- test_matix[, 1]
test.y[test.y[] == "B"] <- 0
test.y[test.y[] == "M"] <- 1

test.y <- to_categorical(test.y) %>% as.array()
test.y <- test.y[, 2]

###Normalizes training/test sets

for(i in 1:20){
  train.x[,i] <- scale(train.x[,i])
}

for(i in 1:20){
  test.x[,i] <- scale(test.x[,i])
}

#Removes used files
rm(model_data, test_matix, train_matix, i, index) 

### Creates sequential model - MLP

ann.32 <- keras_model_sequential() 

weights <- initializer_random_normal(mean = 0, stddev = 0.1, seed = NULL)

ann.32 %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = 20, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

adam <- optimizer_adam(lr = 0.0003)

ann.32 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.32 <- ann.32 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 1
)

plot(history.32)

prediction.32 <- predict_classes(ann.32, test.x) %>% as.matrix

#Creates prediction file
result <- data.frame(prediction.32, test.y) %>%
  write.table("result_32.txt", sep=",") 

# Prints test and train accuracy
test.result.32 <- evaluate(ann.32, test.x, test.y, verbose = 0)
train.result.32 <- evaluate(ann.32, train.x, train.y, verbose = 0)
test.result.32 
########### Experimenting with different activation functions ################

## Sigmoid
ann.sigmoid <- keras_model_sequential() 

ann.sigmoid %>% 
  layer_dense(units = 32, activation = 'sigmoid', input_shape = 20, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.sigmoid %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.sigmoid <- ann.sigmoid %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 1
)

prediction.sigmoid <- predict_classes(ann.sigmoid, test.x) %>% as.matrix

#Creates prediction file
result.sigmoid <- data.frame(prediction.sigmoid, test.y) %>%
  write.table("result_sigmoid.txt", sep=",") 

# Prints test and train accuracy
evaluate(ann.sigmoid, test.x, test.y, verbose = 0)
evaluate(ann.sigmoid, train.x, train.y, verbose = 0)


# Creates table for ploting ReLU/Sigmoid accuracy/loss
relu.sig.loss <- matrix(nrow = 20, ncol = 3)

relu.sig.loss[,1] <- rep(1:20)
relu.sig.loss[,2] <- history.32$metrics$loss
relu.sig.loss[,3] <- history.sigmoid$metrics$loss
colnames(relu.sig.loss) <- c("Epoch", "ReLU","Sigmoid")


plot(relu.sig.loss[,2], type = "o", col = "coral2", ylim = c(0, 1),  
     ylab = "Loss", xlab = "Epoch")

lines(relu.sig.loss[,3], type = "o", col = "blue")
legend(14, 1, c("ReLU", "Sigmoid"), lty = c(1, 1), col = c("Coral2", "Blue"))


relu.sig.acc <- matrix(nrow = 20, ncol = 3)

relu.sig.acc[,1] <- rep(1:20)
relu.sig.acc[,2] <- history.32$metrics$acc
relu.sig.acc[,3] <- history.sigmoid$metrics$acc
colnames(relu.sig.acc) <- c("Epoch", "ReLU","Sigmoid")


plot(relu.sig.acc[,2], type = "o", col = "coral2", ylim = c(0.2, 1),  
     ylab = "Accuracy", xlab = "Epoch")

lines(relu.sig.acc[,3], type = "o", col = "blue")

legend(14, 0.5, c("ReLU", "Sigmoid"), lty = c(1, 1), col = c("Coral2", "Blue"))

########### Experimenting with different number of nodes ################

#### 4 node hidden layer model
ann.4 <- keras_model_sequential() 

ann.4 %>% 
  layer_dense(units = 4, activation = 'relu', input_shape = 20, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.4 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.4 <- ann.4 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.4 <- predict_classes(ann.4, test.x) %>% as.matrix

#Creates prediction file
result.4 <- data.frame(prediction.4, test.y) %>%
  write.table("result_4.txt", sep=",") 

# Prints test and train accuracy
test.result.4 <- evaluate(ann.4, test.x, test.y, verbose = 0)
train.result.4 <- evaluate(ann.4, train.x, train.y, verbose = 0)


#### 8 node hidden layer model
ann.8 <- keras_model_sequential() 

ann.8 %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = 20, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.8 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.8 <- ann.8 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.8 <- predict_classes(ann.8, test.x) %>% as.matrix

#Creates prediction file
result.8 <- data.frame(prediction.8, test.y) %>%
  write.table("result_8.txt", sep=",") 

# Prints test and train accuracy
test.result.8 <- evaluate(ann.8, test.x, test.y, verbose = 0)
train.result.8 <- evaluate(ann.8, train.x, train.y, verbose = 0)


#### 16 node hidden layer model
ann.16 <- keras_model_sequential() 

ann.16 %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = 20, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.16 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.16 <- ann.16 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.16 <- predict_classes(ann.16, test.x) %>% as.matrix

#Creates prediction file
result.16 <- data.frame(prediction.16, test.y) %>%
  write.table("result_16.txt", sep=",") 

# Prints test and train accuracy
test.result.16 <- evaluate(ann.16, test.x, test.y, verbose = 0)
train.result.16 <- evaluate(ann.16, train.x, train.y, verbose = 0)


#### 64 node hidden layer model
ann.64 <- keras_model_sequential() 

ann.64 %>% 
  layer_dense(units = 64, activation = 'relu', input_shape = 20, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.64 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.64 <- ann.64 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.64 <- predict_classes(ann.64, test.x) %>% as.matrix

#Creates prediction file
result.64 <- data.frame(prediction.64, test.y) %>%
  write.table("result_64.txt", sep=",") 

# Prints test and train accuracy
test.result.64 <- evaluate(ann.64, test.x, test.y, verbose = 0)
train.result.64 <- evaluate(ann.64, train.x, train.y, verbose = 0)

table(prediction.64, test.y)

#Creates bar plot for training data
nodes.train <- matrix(ncol = 2, nrow = 5)
colnames(nodes.train) <- c("Nodes","Accuracy")
nodes.train[,1] <- c("4","8","16","32","64")

nodes.train[,2] <- c(round(train.result.4$acc, 2)*100,
                     round(train.result.8$acc, 2)*100,
                     round(train.result.16$acc, 2)*100,
                     round(train.result.32$acc, 2)*100,
                     round(train.result.64$acc, 2)*100)

nodes.train <- data.frame(nodes.train)
nodes.train$Nodes <- factor(nodes.train$Nodes, c(4, 8, 16, 32, 64))

ggplot(nodes.train, aes(x = Nodes, y = Accuracy)) +
  geom_bar(stat = "identity")+
  ggtitle("Training Accuracy")

# Creates bar plot for testing data
nodes.test <- matrix(ncol = 2, nrow = 5)
colnames(nodes.test) <- c("Nodes","Accuracy")
nodes.test[,1] <- c("4","8","16","32","64")

nodes.test[,2] <- c(round(test.result.4$acc, 2)*100,
                    round(test.result.8$acc, 2)*100,
                    round(test.result.16$acc, 2)*100,
                    round(test.result.32$acc, 2)*100,
                    round(test.result.64$acc, 2)*100)

nodes.test <- data.frame(nodes.test)
nodes.test$Nodes <- factor(nodes.test$Nodes, c(4, 8, 16, 32, 64))

ggplot(nodes.test, aes(x = Nodes, y = Accuracy)) +
  geom_bar(stat = "identity") +
  ggtitle("Testing Accuracy")


### prints confusion matrices 
# Confusion Matrix results of all models
table(prediction.4, test.y)
table(prediction.8, test.y)
table(prediction.16, test.y)
table(prediction.32, test.y)
table(prediction.64, test.y)
table(prediction.sigmoid, test.y)


#Applying Semi- Supervised Learning
#closing previous graphs
graphics.off()

#Load Library
library(ssc)
library(kernlab)
library(C50)

#Create semi-supervised dataset
model_data <- cbind (diagnosis=cancer_data$diagnosis, cancer_data_wcor)
cls <- which(colnames(model_data) == "diagnosis")
x <- model_data[, -cls] # instances without classes
y <- model_data[, cls] # the classes
x <- scale(x) # scale the attributes for distance calculations
set.seed(3)

#Use 70% of instances for training
tra.idx <- sample(x = length(y), size = ceiling(length(y) * 0.7))
xtrain <- x[tra.idx,] # training instances
ytrain <- y[tra.idx] # classes of training instances

# Use 70% of train instances as unlabeled set
tra.na.idx <- sample(x = length(tra.idx), size = ceiling(length(tra.idx) * 0.7))
ytrain[tra.na.idx] <- NA #remove class of unlabeled instances

# Use the other 30% of instances for inductive test
tst.idx <- setdiff(1:length(y), tra.idx)
xitest <- x[tst.idx,] # test instances
yitest <- y[tst.idx] # classes of instances in xitest

# Use the unlabeled examples for transductive test
xttest <- x[tra.idx[tra.na.idx],] # transductive test instances
yttest <- y[tra.idx[tra.na.idx]] # classes of instances in xttest


dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE))
ditest <- as.matrix(proxy::dist(x = xitest, y = xtrain, method = "euclidean", by_rows = TRUE))

ktrain <- as.matrix(exp(- 0.048 * dtrain^2))
kitest <- as.matrix(exp(- 0.048 * ditest^2))

m.selft <- selfTraining(x = ktrain, y = ytrain, x.inst = FALSE, learner = ksvm,
                        learner.pars = list(kernel = "matrix", prob.model = TRUE), pred = function(m, k)
                          predict(m, as.kernelMatrix(k[, SVindex(m)]), type = "probabilities"))

m.snnrce <- snnrce(x = xtrain, y = ytrain, dist = "Euclidean")
m.setred <- setred(x = xtrain, y = ytrain, dist = "Euclidean", learner = ksvm, 
                   learner.pars = list(prob.model = TRUE), pred = predict,
                   pred.pars = list(type = "probabilities"))
m.trit <- triTraining(x = xtrain, y = ytrain, learner = ksvm, learner.pars = list(prob.model = TRUE),
                      pred = predict, pred.pars = list(type = "probabilities"))
m.cobc <- coBC(x = xtrain, y = ytrain, N = 5, learner = ksvm, learner.pars = list(prob.model = TRUE), 
               pred = predict, pred.pars = list(type = "probabilities"))

m.demo <- democratic(x = xtrain, y = ytrain, learners = list(knn3, ksvm, C5.0),
                     learners.pars = list(list(k=1), list(prob.model = TRUE), NULL),
                     preds = list(predict, predict, predict), preds.pars =
                       list(NULL, list(type = "probabilities"), list(type = "prob")))

l1nn <- function(indexes, cls){
  m <- oneNN(y = cls)
  attr(m, "tindexes") <- indexes
  m
}
l1nn.prob <- function(m, indexes) {
  predict(m, dtrain[indexes, attr(m, "tindexes")], type = "prob")
}
lsvm <- function(indexes, cls){
  m = ksvm(ktrain[indexes, indexes], cls, kernel = "matrix", prob.model = TRUE)
  attr(m, "tindexes") <- indexes[SVindex(m)]
  m
}
lsvm.prob <- function(m, indexes) {
  k <- as.kernelMatrix(ktrain[indexes, attr(m, "tindexes")])
  predict(m, k, type = "probabilities")
}
m.demoG <- democraticG(y = ytrain, gen.learners = list(l1nn, lsvm), gen.preds = list(l1nn.prob, lsvm.prob))

p.selft <- predict(m.selft, as.kernelMatrix(kitest[, m.selft$instances.index]))

p.selfttransd <- predict(m.selft, as.kernelMatrix(ktrain[tra.na.idx, m.selft$instances.index]))

p.snnrce <- predict(m.snnrce, xitest)
p.setred <- predict(m.setred, xitest)
p.trit <- predict(m.trit, xitest)
p.cobc <- predict(m.cobc, xitest)

p.demo <- predict(m.demo, xitest)

m1.pred1 <- predict(m.demoG$model[[1]], ditest[, m.demoG$model.index[[1]]], type ="class")
m1.pred2 <- predict(m.demoG$model[[2]], as.kernelMatrix(kitest[, m.demoG$model.index[[2]]]))
p.demoG <- democraticCombine(pred = list(m1.pred1, m1.pred2), m.demoG$W, m.demoG$classes)


p <- list(p.selft, p.snnrce, p.setred, p.trit, p.cobc, p.demo)
acc <- sapply(X = p, FUN = function(i) {caret::confusionMatrix(table(i, yitest))$overall[1]})

names(acc) <- c("SelfT","SNNRCE","SETRED","TriT", "coBC","Demo")
acc
barplot(acc, beside = T, ylim = c(0.80,1), xpd = FALSE, las = 2,
        col=rainbow(n = 6, start = 3/6, end = 4/6, alpha = 0.6), ylab = "Accuracy")

labeled.idx <- which(!is.na(ytrain))# indices of the initially labeled instances
xilabeled <- xtrain[labeled.idx,] # labeled instances
yilabeled <- ytrain[labeled.idx] # related classes

svmBL <- ksvm(x = xilabeled, y = yilabeled, prob.model = TRUE) # build SVM
p.svmBL <- predict(object = svmBL, newdata = xitest) # classify with SVM
table_mat <- table(yitest, p.svmBL)
acc.svm <- sum(diag(table_mat)) / sum(table_mat)
acc.svm
abline(h = caret::confusionMatrix(table(p.svmBL, yitest))$overall[1], col = "red", lwd = 2)
legend(x = 2, y = 1.0, col = c("red"), legend=c("Base line"), lty = 1, lwd = 2)


#Unsupervised
graphics.off()

# Calculate the (Euclidean) distances
data_dist <- dist(cancer_data_wcor)

# Create a hierarchical clustering model
bc_hclust <- hclust(data_dist, method = "complete")

# Plot hierarchical clustering model
plot(bc_hclust)

# Cut tree so that it has 4 clusters
bc_hclust_cluster <- cutree(bc_hclust, k = 4)

# Compare cluster membership to actual diagnoses
table(bc_hclust_cluster, diagnosis)

# count out of place observations based on cluster 
# basically just summing the row mins here
sum(apply(table(bc_hclust_cluster, diagnosis), 1, min))

#K-means clustering
bc_km <- kmeans(data_dist, centers = 2, nstart = 20)

# Compare k-means to actual diagnosis
table(bc_km$cluster, diagnosis)
sum(apply(table(bc_km$cluster, diagnosis), 1, min))

# Compare k-means to hierarchical clustering
table(bc_hclust_cluster, bc_km$cluster)
sum(apply(table(bc_hclust_cluster, bc_km$cluster), 1, min))









