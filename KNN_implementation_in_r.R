# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
library(caret)
library(class)
library(caTools)
library(psych)

wine <- read.csv("wine.csv")

describe(wine)

# Data Preprocessing

# Removal of the ID column
wine <- subset(wine, select = -c(Id))
# Quality column is converted to factor from integer.
wine$quality <- as.factor(wine$quality)

set.seed(255)

# Train-Test Split
split <- sample.split(wine$quality, SplitRatio = 0.7)
train <- subset(wine, split == TRUE)
test <- subset(wine, split == FALSE)

# # Feature Scaling
# (x - mean(x)) / sd(x) which is Z-Score
train_scaled <- scale(train[-12])
test_scaled <- scale(test[-12])

# Square Root of N to find out the best K value
n <- nrow(train)
best_k <- round(sqrt(n))
print(best_k)

# KNN Model
test_pred <- knn(train = train_scaled, 
                 test = test_scaled,
                 cl = train$quality, 
                 k = best_k)

# Metrics
actual <- test$quality
cm <- confusionMatrix(as.factor(test_pred), actual)
print(cm)
