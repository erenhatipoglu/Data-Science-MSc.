library(lmtest)
library(olsrr)
library(ISLR)
library(psych)
library(corrplot)
library(leaps)
library(MASS)
library(car)
  
data("Carseats")
head(Carseats)
set.seed(2023900089)
sample_size <- 0.7 * nrow(Carseats)
  
index <- sample(nrow(Carseats), size=sample_size, replace=F)
  
train_data <- Carseats[index, ]
test_data <- Carseats[-index, ]
  
head(train_data)
head(test_data)

# Descriptive Statistics
str(train_data)

describe(train_data)
  
par(mfrow=c(2,2))
hist(train_data$Sales, main="Satış Dağılımı", col="cornsilk", xlab="Sales")
boxplot(train_data$Sales, main="Satış Dağılımı", col="grey", xlab="Sales", horizontal=T)
hist(train_data$CompPrice, main="CompPrice Dağılımı", col="maroon", xlab="CompPrice")
hist(train_data$Price, main="Price Dağılımı", col="lightgreen", xlab="Price")
par(mfrow=c(1,1))
  
# Matrix Plot
pairs(train_data, lower.panel = NULL)
  
# Multiple Linear Regression
lm_model <- lm(Sales ~ ., data = train_data)
summary(lm_model)
  
# Variance Inflation Factor (the amount of multicollinearity)
vif_values <- vif(lm_model)
vif_values


# Best Subsets Regression
best1 <- ols_step_all_possible(lm_model)
summary(best1)

# The highest CP value (m1) and the hightest Adjusted-R2 (m2) subsets are selected.
model1 <- lm(Sales ~ CompPrice + Income + Advertising + Price + ShelveLoc + Age + Education, data = train_data)
model2 <- lm(Sales ~ CompPrice + Income + Advertising + Price + ShelveLoc + Age, data = train_data)

# PRESS (Prediction Error Sum of Squares)
press_model1 <- sum((test_data$Sales - pred_model1)^2)
press_model2 <- sum((test_data$Sales - pred_model2)^2)

# RMSE (Root Mean Square Error)
rmse_model1 <- sqrt(mean((test_data$Sales - pred_model1)^2))
rmse_model2 <- sqrt(mean((test_data$Sales - pred_model2)^2))

# MAE (Mean Absolute Error)
mae_model1 <- mean(abs(test_data$Sales - pred_model1))
mae_model2 <- mean(abs(test_data$Sales - pred_model2))

press_model1
press_model2
rmse_model1
rmse_model2
mae_model1
mae_model2

# We continue with the model1 since it performs better

# Residuals of the selected model (model1)
residuals_model1 <- residuals(model1)

par(mfrow=c(1,2))
# Histogram of residuals
hist(residuals_model1, main="Histogram of Residuals", col="lightblue", xlab="Residuals")

# Q-Q plot
qqnorm(residuals_model1, pch =2)
qqline(residuals_model1, col = "darkred", lwd = 2)
par(mfrow=c(1,1))

# Shapiro-Wilk test for normality
shapiro_test <- shapiro.test(residuals_model1)
shapiro_test

#Shapiro-Wilk test indicates the residuals are normally distributed.
par(mfrow=c(2,2))
plot(model1, which = 1)
par(mfrow=c(1,1))

# Scale-Location Plot ve Breusch-Pagan test for homoscedasticity
plot(model1, which = 1)
bp <- bptest(model1)
bp

# Residuals are nonconstant varianced.

# Cook's Distance and Boxplot for the detection of outliers
plot(model1, which = 4)
boxplot(residuals_model1, horizontal=T)

# Residuals vs Leverage to see the influential observations
plot(model1, which = 5)

## Outliers are removed to optimize the model

# Calculate the Interquartile Range (IQR) of residuals
residuals_iqr <- IQR(residuals_model1)

# Define the lower and upper bounds for outliers
lower_bound <- quantile(residuals_model1)[2] - 1.3 * residuals_iqr
upper_bound <- quantile(residuals_model1)[4] + 1.3 * residuals_iqr

# Identify outliers
outliers <- residuals_model1 < lower_bound | residuals_model1 > upper_bound

# Exclude outliers from the dataset
cleaned_data <- train_data[!outliers, ]

# Fit the model again with the cleaned data
cleaned_model <- lm(Sales ~ CompPrice + Income + Advertising + Price + ShelveLoc + Age + Education, data = cleaned_data)

summary(cleaned_model)

# Plot the boxplot of residuals for the cleaned model
residuals_cleaned <- residuals(cleaned_model)
boxplot(residuals_cleaned, horizontal = TRUE)

final_model <- cleaned_model
bptest(final_model)

summary(model1)
summary(final_model) #Final model (with removed outliers) has a better R2 and F values.

# New prediction for a new observation

new_observation <- data.frame(
  CompPrice = 120,
  Income = 90,
  Advertising = 10,
  Population = 200,
  Price = 120,
  ShelveLoc = "Good",
  Age = 35,
  Education = 15,
  Urban = "Yes",
  US = "Yes",
  Sales = NA  # Sales is the predicted value.
)

prediction_interval <- predict(final_model, newdata = new_observation, interval = "prediction", level = 0.95)
prediction_interval

