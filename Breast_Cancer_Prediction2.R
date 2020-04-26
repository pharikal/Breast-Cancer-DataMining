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
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv"

# Read the data
cancer_data <- read.csv(url, sep = ",", header = TRUE, 
                        stringsAsFactors=FALSE) %>% as.data.frame

colnames(cancer_data)

#Data Understading
head(cancer_data)
str(cancer_data)
cancer_data1 <- cancer_data
cancer_data$Classification[cancer_data$Classification == "1"] <- "N"
cancer_data$Classification[cancer_data$Classification == "2"] <- "Y"

#Check if the dataset has any missing value
map(cancer_data, function(.x) sum(is.na(.x)))

#Check data proportions 
classify.table <- table(cancer_data$Classification)
prop.table(classify.table)

#Plot the data distribution using a pie chart
colors <- terrain.colors(2)
classify.prop.table <- prop.table(classify.table)*100
classify.prop.df <- as.data.frame(classify.prop.table)
pielabels <- sprintf("%s - %3.1f%s", classify.prop.df[,1], classify.prop.table, "%")
pie(classify.prop.table,
    labels=pielabels,  
    clockwise=TRUE,
    col=colors,
    border="gainsboro",
    radius=0.8,
    cex=0.8, 
    main="frequency of cancer Classification")
legend(1, .4, legend=classify.prop.df[,1], cex = 0.7, fill = colors)



#Plot histograms of all attributes by Classification
ggplot(data = melt(cancer_data, id.var = "Classification"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill= Classification), alpha=0.5) + facet_wrap(~variable, scales ='free_x')


#Check if the is any correlation between variables as machine learning algorithms 
#assume that the predictor variables are independent from each others.

# Correlation Matrix 
correlationMatrix <- cor(cancer_data[,1:9]) 
corrplot(correlationMatrix, order = "hclust", tl.cex = 0.5, addrect = 8)

# Attributes that are highly corrected (ideally >0.90)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9) 
# Indices of highly correlated attributes 
highlyCorrelated

# Remove correlated variables 
cancer_data_wcor <- cancer_data[,1:9] %>%select(-highlyCorrelated) 
# Check column count after removing correlated variables 
ncol(cancer_data_wcor)
colnames(cancer_data_wcor)

#Data Modelling
# Plot of PCA 
pca_res_data1 <- prcomp(cancer_data[,1:9], center = TRUE, scale = TRUE) 
plot(pca_res_data1, type="l", main = "")
grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()
summary(pca_res_data1)
fviz_pca_biplot(pca_res_data1, col.ind = cancer_data$Classification, col="black",
                palette = "jco", geom = "point", repel=TRUE,
                legend.title="Diagnosis", addEllipses = TRUE)


# Plot of PCA
pca_res_data2 <- prcomp(cancer_data_wcor, center = TRUE, scale = TRUE) 
plot(pca_res_data2, type="l", main = "")
grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()
summary(pca_res_data2)
fviz_pca_biplot(pca_res_data2, col.ind = cancer_data$Classification, col="black",
                palette = "jco", geom = "point", repel=TRUE,
                legend.title="Diagnosis", addEllipses = TRUE)

#Compare plot of Diagnosis between PC1 and PC2
pca_df <- as.data.frame(pca_res_data2$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=cancer_data$Classification)) + geom_point(alpha=0.5)

g_pc1 <- ggplot(pca_df, aes(x=PC1, fill=cancer_data$Classification)) +
  geom_density(alpha=0.25) 
g_pc2 <- ggplot(pca_df, aes(x=PC2, fill=cancer_data$Classification)) +
  geom_density(alpha=0.25)
grid.arrange(g_pc1, g_pc2, ncol=2)

#Let's visuzalize the first 3 components.
df_pcs <- cbind(as_tibble(cancer_data$Classification), as_tibble(pca_res_data2$x))
GGally::ggpairs(df_pcs, columns = 2:4, ggplot2::aes(color = value))


#Linear Discriminant Analysis (LDA) 
lda_res_data <- MASS::lda(Classification~., data = cancer_data, center = TRUE, scale = TRUE) 
lda_res_data

# Data frame of the LDA for visualization purposes 
lda_df_predict <- predict(lda_res_data, cancer_data)$x %>% 
                      as.data.frame %>% cbind(Classification=cancer_data$Classification)

# LDA Plot 
ggplot(lda_df_predict, aes(x=LD1, fill=Classification)) + geom_density(alpha=0.5) + 
  xlab("LD1") + ylab("Density") + labs(fill = "Classification")

#Data Preparation
# Preparing Train and Test datasets 
set.seed(1815)

model_data <- cbind (Classification=cancer_data$Classification, cancer_data_wcor)

#To Draw boxplots for understanding the outliers of the data
boxplot(model_data)

#To Draw Bivariate Boxplot against classificatin
cancer_data1 <- cancer_data1[, c(-5)]
library(MVA)
par(mfrow = c(1,3))
bvbox(cancer_data1[,c(1,9)], xlab = "Classification", ylab = "Age")
bvbox(cancer_data1[,c(2,9)], xlab = "Classification", ylab = "BMI")
bvbox(cancer_data1[,c(3,9)], xlab = "Classification", ylab = "Glucose")
bvbox(cancer_data1[,c(4,9)], xlab = "Classification", ylab = "Insulin")
bvbox(cancer_data1[,c(5,9)], xlab = "Classification", ylab = "Leptin")
bvbox(cancer_data1[,c(6,9)], xlab = "Classification", ylab = "Adiponectin")
bvbox(cancer_data1[,c(7,9)], xlab = "Classification", ylab = "Resistin")
bvbox(cancer_data1[,c(8,9)], xlab = "Classification", ylab = "MCP.1")

#Outlier Reduction as Glucose, Insulin, Leptin, Adiponectin, Resistin, MCP.1 has outliers
dim(model_data)
hull1 <- chull(model_data[,c(1,4)])
outlier <- model_data[-hull1,]


hull2 <- chull(outlier[,c(1,5)])
model_data <- outlier[-hull2,]

hull3 <- chull(model_data[,c(1,6)])
outlier <- model_data[-hull3,]

hull4 <- chull(outlier[,c(1,7)])
model_data <- outlier[-hull4,]

hull5 <- chull(model_data[,c(1,8)])
outlier <- model_data[-hull5,]

hull6 <- chull(outlier[,c(1,9)])
model_data <- outlier[-hull6,]

dim(model_data)

#Appliying Semi- Supervised Learning
#closing brevious graphs
graphics.off()

#Load Library
library(ssc)
library(kernlab)
library(C50)

#Create semi-supervised dataset
model_data <- cbind (Classification=cancer_data$Classification, cancer_data_wcor)
cls <- which(colnames(model_data) == "Classification")
x <- model_data[, -cls] # instances without classes
y <- model_data[, cls] # the classes
x <- scale(x) # scale the attributes for distance calculations
set.seed(3)

#Use 80% of instances for training
tra.idx <- sample(x = length(y), size = ceiling(length(y) * 0.8))
xtrain <- x[tra.idx,] # training instances
ytrain <- y[tra.idx] # classes of training instances

# Use 40% of train instances as unlabeled set
tra.na.idx <- sample(x = length(tra.idx), size = ceiling(length(tra.idx) * 0.4))
ytrain[tra.na.idx] <- NA #remove class of unlabeled instances

# Use the other 50% of instances for inductive test
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
barplot(acc, beside = T, ylim = c(0.20,1), xpd = FALSE, las = 2,
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


