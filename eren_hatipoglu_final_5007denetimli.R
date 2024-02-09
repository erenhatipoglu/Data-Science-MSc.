library(boot)
library(randomForest)
library(tree)
library(ROCR)
library(caret)
library(MASS)
library(MASS)  
library(caret)  
library(klaR)  
library(heplots)

abalone_data <- read.csv("C:\\Users\\erenh\\OneDrive\\lenova\\deu data science YL\\denetimli öğrenme\\sınav2\\abalone_veriseti.data")
veriseti_açıklama <- read.csv("C:\\Users\\erenh\\OneDrive\\lenova\\deu data science YL\\denetimli öğrenme\\sınav2\\abalone_açıklama.names")

# Assign column names (based on standard abalone dataset attributes)
colnames(abalone_data) <- c("Sex", "Length", "Diameter", "Height", "WholeWeight",
                       "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings")

# Encode the categorical variable 'Sex'
abalone_data$Sex <- as.factor(abalone_data$Sex)

# Splitting the dataset into features (X) and target (y)
X <- abalone_data[, -which(names(abalone_data) == "Rings")]
y <- abalone_data$Rings

# Set the seed for reproducibility (replace '123' with the last three digits of your student number)
set.seed(089)

# Splitting the data into training and test sets (70-30 split)
n_train <- round(nrow(X) * 0.7)
index <- sample(1:nrow(X), n_train)
X_train <- X[index, ]
y_train <- y[index]
X_test <- X[-index, ]
y_test <- y[-index]

########### Part A.

# Linear Regression (LM)
lm_model <- lm(y_train ~ ., data = X_train)
summary(lm_model)
plot(lm_model, which=1)
# Insignificant variables in linear regression will be removed

X_train_copy <- X_train
y_train_copy <- y_train
X_test_copy <- X_test
y_test_copy <- y_test

# Removing the insignificant variables
X_train_copy$Sex_NoM <- factor(ifelse(X_train_copy$Sex == "M", NA, as.character(X_train_copy$Sex)),
                               levels = levels(X_train_copy$Sex)[levels(X_train_copy$Sex) != "M"])

lm_model2 <- lm(y_train_copy ~ . - Length - Sex, data = X_train_copy)

summary(lm_model2)

# Application of linear regression to the test set

X_train_copy$Sex <- factor(X_train_copy$Sex, levels = levels(X_train_copy$Sex)
                           [levels(X_train_copy$Sex) != "M"])
X_test_copy$Sex <- factor(X_test_copy$Sex, levels = levels(X_train_copy$Sex))

lm_model2_test <- lm(y_test_copy ~ . - Length - Sex, data = X_test_copy)

# Summary of the model applied to the test data
summary(lm_model2_test)


# Regression Tree
rt_model <- tree(y_train ~ ., data = X_train)
summary(rt_model)
plot(rt_model)
text(rt_model, pretty=0)
summary(rt_model)

# Cross-Validation in Regression Tree
cv_abalone <- cv.tree(rt_model)
plot(cv_abalone$size, cv_abalone$dev, type = "b", xlab = "Tree Size", ylab = "CV Sapma")

# Pruning the regression tree
prune_rt_model <- prune.tree(rt_model, best = 9)
plot(prune_rt_model)
text(prune_rt_model, pretty = 0)

summary(prune_rt_model)

yhat_original <- predict(rt_model, newdata = X_test)
rmse_original <- sqrt(mean((yhat_original - y_test)^2))
rmse_original

yhat_pruned <- predict(prune_rt_model, newdata = X_test)
rmse_pruned <- sqrt(mean((yhat_pruned - y_test)^2))
rmse_pruned

# Application of regression tree on test data
yhat <- predict(prune_rt_model, newdata = X_test)
# Extract the actual response values from the test data
rt_test_actual <- y_test
# Plot the predicted values against the actual values
plot(yhat, rt_test_actual)
abline(0, 1)
sqrt(mean((yhat - rt_test_actual)^2))


# Regression Tree with Bagging (Bootstrap Aggregation)
brt_model <- randomForest(y_train ~ ., data = X_train, mtry=8, importance=TRUE)
brt_model

brt_pred_train <- predict(brt_model, X_train)
par(mfrow=c(1,2))
plot(brt_pred_train, y_train)
abline(0, 1)
title("bagging train")

sqrt(mean((brt_pred_train - y_train)^2))

# Using the bagging model on test data
yhat_bag <- predict(brt_model, newdata = X_test)


bag_test_actual <- y_test
plot(yhat_bag, bag_test_actual)
abline(0, 1)
title("bagging test")
par(mfrow=c(1,1))
sqrt(mean((yhat_bag - bag_test_actual)^2))

brt_model$importance
varImpPlot(brt_model)


# Random Forest Regression
rfr_model <- randomForest(y_train ~ ., data = X_train, mtry=2.6, importance=TRUE)
rfr_model

# Plots
rfr_pred_train <- predict(rfr_model, X_train)
par(mfrow=c(1,2))

plot(rfr_pred_train, y_train)
abline(0, 1)
title("random forest train set")

# Train RMSE
sqrt(mean((rfr_pred_train - y_train)^2))

# Application of random forest on the test data set
rfr_pred <- predict(rfr_model, X_test)

plot(rfr_pred, y_test)
abline(0, 1)
title("random forest test set")
par(mfrow=c(1,1))

# Test RMSE
sqrt(mean((rfr_pred - y_test)^2))

rfr_model$importance
varImpPlot(rfr_model)


########### Part B.

# Sex = I filtering
infant_abalone <- subset(abalone_data, Sex == 'I')

# With the median (8) as the threshold value, the Rings variable is truncated based on Sex=Infant
infant_abalone$BinaryRings <- ifelse(infant_abalone$Rings >= 8, 1, 0)

# Two-Level coding (transforming to factor)
infant_abalone$Sex <- as.factor(infant_abalone$Sex)

# Sorting the dataset for part B
X <- infant_abalone[, -which(names(infant_abalone) %in% c("Rings", "BinaryRings"))]
y <- infant_abalone$BinaryRings

set.seed(089)

# train test split
index <- sample(1:nrow(X), nrow(X)*0.7)
X_train <- X[index, ]
X_test <- X[-index, ]
y_train <- y[index]
y_test <- y[-index]




y_train <- as.factor(y_train)
y_test <- as.factor(y_test)

# Classification Tree
ct_model <- tree(y_train ~ ., data = X_train)
ct_pred <- predict(ct_model, X_test, type = "class")
summary(ct_model)

conf_matrix <- table(ct_pred, y_test)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
accuracy
plot(ct_model)
text(ct_model, pretty=0)

# Pruning the Classification Tree (CT)
cv.ct_model <- cv.tree(ct_model, FUN=prune.misclass)

plot(cv.ct_model$size, cv.ct_model$dev, type="b")

pruned_ct_model <- prune.misclass(ct_model, best=2)

plot(pruned_ct_model)
summary(pruned_ct_model)
text(pruned_ct_model, pretty=0)

# Prediction on test set with pruned tree
pruned_ct_pred <- predict(pruned_ct_model, X_test, type="class")
conf_matrix_pruned <- table(pruned_ct_pred, y_test)
accuracy_pruned <- sum(diag(conf_matrix_pruned)) / sum(conf_matrix_pruned)
accuracy_pruned


# Bagging with Classification Tree (BCT)
bct_model <- randomForest(y_train ~ ., data = X_train, mtry = 8, importance=TRUE)
bct_pred_train <- predict(bct_model, X_train)
bct_pred_test <- predict(bct_model, X_test)

# train set accuracy
train_accuracy <- mean(bct_pred_train == y_train)
train_accuracy
importance(bct_model)
varImpPlot(bct_model)

# test set accuracy
test_accuracy <- mean(bct_pred_test == y_test)
test_accuracy


# Random Forest Classification (RFC)
rfc_model <- randomForest(y_train ~ ., data = X_train, mtry = 2.6, importance = TRUE)

# train accuracy
rfc_pred_train <- predict(rfc_model, X_train)
accuracy_train <- sum(rfc_pred_train == y_train) / length(y_train)
accuracy_train

# test accuracy
rfc_pred_test <- predict(rfc_model, X_test)
accuracy_test <- sum(rfc_pred_test == y_test) / length(y_test)
accuracy_test

importance(rfc_model)
varImpPlot(rfc_model)


# Save 'Sex' columns before excluding it for Logistic Regression
Sex_train <- X_train$Sex
Sex_test <- X_test$Sex

# Exclude 'Sex' variable for Logistic Regression
X_train_no_sex <- X_train[, !(names(X_train) == 'Sex')]
X_test_no_sex <- X_test[, !(names(X_test) == 'Sex')]

# Logistic Regression (LR)
lr_model <- glm(y_train ~ ., data = X_train_no_sex, family = "binomial")
lr_pred <- predict(lr_model, X_test_no_sex, type = "response")
lr_pred_class <- ifelse(lr_pred > 0.5, 1, 0)
summary(lr_model)

# Confusion Matrix for Training Set
lr_pred_train <- predict(lr_model, X_train_no_sex, type = "response")
lr_pred_class_train <- ifelse(lr_pred_train > 0.5, 1, 0)
confusion_lr_train <- confusionMatrix(factor(lr_pred_class_train), factor(y_train))
print("Confusion Matrix for Training Set:")
print(confusion_lr_train)

# acc train
accuracy_train <- sum(lr_pred_class_train == y_train) / length(y_train)
accuracy_train

# Confusion Matrix for Test Set
confusion_lr <- confusionMatrix(factor(lr_pred_class), factor(y_test))
print("Confusion Matrix for Test Set:")
print(confusion_lr)

# acc test
accuracy_test <- sum(lr_pred_class == y_test) / length(y_test)
accuracy_test


# Plot related to logistic regression
par(mfrow=c(2,2))
plot(lr_model)
par(mfrow=c(1,1))

# Sensitivity and Specificity
p.tahmin.chd <- fitted(lr_model)  
p.tahmin.chd[p.tahmin.chd > 0.5] <- 1
p.tahmin.chd[p.tahmin.chd <= 0.5] <- 0
confusion <- table(p.tahmin.chd, y_train)
sensitivity <- confusion[2, 2] / sum(confusion[, 2])  # Sensitivity
specificity <- confusion[1, 1] / sum(confusion[, 1])  # Specificity
# Print Sensitivity and Specificity
print(paste("Sensitivity(Recall):", sensitivity))
print(paste("Specificity:", specificity))


# Linear Discriminant Analysis

# Add 'Sex' variable back for subsequent analyses
X_train$Sex <- Sex_train
X_test$Sex <- Sex_test

# Check for variables with constant or very little variation
constant_vars <- apply(X_train, 2, function(x) length(unique(x)) == 1)
# Remove those variables
X_train_no_const <- X_train[, !constant_vars]
X_test_no_const <- X_test[, !constant_vars]

lda_model <- lda(y_train ~ ., data = X_train_no_const)
lda_pred <- predict(lda_model, X_test_no_const)$class
lda_model

lda_model$prior

#acc train
lda_pred_train <- predict(lda_model, X_train_no_const)$class
accuracy_lda_train <- mean(lda_pred_train == y_train)
accuracy_lda_train

# Evaluation for LDA
cfmatrix_lda <- table(Predicted = lda_pred, Actual = y_test)
cfmatrix_lda

# acc test
accuracy_lda <- sum(diag(cfmatrix_lda)) / sum(cfmatrix_lda)
accuracy_lda

# Visualization of the LDA partition
partimat(y_train ~ ., data = X_train_no_const, method = "lda")
partimat(y_train ~ ., data = X_train_no_const, method = "lda", plot.matrix = TRUE, imageplot = FALSE)


# Quadratic Discriminant Analysis

# QDA Model
qda_model <- qda(y_train ~ ., data = X_train_no_const)
qda_pred <- predict(qda_model, X_test)$class
qda_model

qda_model$prior

#acc train
qda_pred_train <- predict(qda_model, X_train_no_const)$class
accuracy_qda_train <- mean(qda_pred_train == y_train)
accuracy_qda_train

# Evaluation for QDA
cfmatrix_qda <- table(Predicted = qda_pred, Actual = y_test)
cfmatrix_qda

# acc test
accuracy_qda <- sum(diag(cfmatrix_qda)) / sum(cfmatrix_qda)
accuracy_qda


# Covariance Matrix
box_m <- boxM(X_train_no_const, y_train)
print(box_m)

cov_ellipses <- covEllipses(X_train_no_const, group = y_train, variables = 1:7)


### Receiver Operating Characteristic Curve (ROC)

# Logistic Regression - Get predicted probabilities
lr_probs <- predict(lr_model, newdata = X_test, type = "response")

# Random Forest Classification - Get predicted probabilities for the positive class
# Extract the 'Sex' column from the original dataset for the test indices
sex_test <- X[-index, "Sex"]

# Add the 'Sex' variable back to the test set for Random Forest predictions
X_test_with_sex <- cbind(X_test, Sex = sex_test)

# Get the predicted probabilities using the updated test set
rfc_probs <- predict(rfc_model, newdata = X_test_with_sex, type = "prob")[,2]

# Linear Discriminant Analysis - Get posterior probabilities for the positive class
lda_probs <- predict(lda_model, newdata = X_test)$posterior[,2]

# Quadratic Discriminant Analysis - Get posterior probabilities for the positive class
qda_probs <- predict(qda_model, newdata = X_test)$posterior[,2]

lr_pred <- prediction(lr_probs, y_test)
rfc_pred <- prediction(rfc_probs, y_test)
lda_pred <- prediction(lda_probs, y_test)
qda_pred <- prediction(qda_probs, y_test)

perf_lr <- performance(lr_pred, measure = "tpr", x.measure = "fpr")
perf_rfc <- performance(rfc_pred, measure = "tpr", x.measure = "fpr")
perf_lda <- performance(lda_pred, measure = "tpr", x.measure = "fpr")
perf_qda <- performance(qda_pred, measure = "tpr", x.measure = "fpr")

# Plot ROC
plot(perf_lr, col = "blue", type = "b", pch = 1, main = "ROC Curves")
plot(perf_rfc, col = "red", type = "b", pch = 2, add = TRUE)
plot(perf_lda, col = "green", type = "b", pch = 3, add = TRUE)
plot(perf_qda, col = "purple", type = "b", pch = 4, add = TRUE)

# Add legend
legend("bottomright", legend = c("LR", "RFC", "LDA", "QDA"), 
       col = c("blue", "red", "green", "purple"), 
       lty = 1, pch = 1:4)


# Area Under the Curve (AUC)
auc_lr <- performance(lr_pred, measure = "auc")@y.values[[1]]
auc_rfc <- performance(rfc_pred, measure = "auc")@y.values[[1]]
auc_lda <- performance(lda_pred, measure = "auc")@y.values[[1]]
auc_qda <- performance(qda_pred, measure = "auc")@y.values[[1]]

auc_lr
auc_rfc
auc_lda
auc_qda

