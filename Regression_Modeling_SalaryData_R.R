salary <- read.csv("C:/Users/erenh/OneDrive/lenova/deu data science YL/denetimli öğrenme/ödev1/Salary.csv")
salary

install.packages("plotly")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("dplyr")
install.packages("car")
install.packages("ISLR")
install.packages("leaps")
install.packages("caret")

# Removing the problematic rows
rows_to_del <- c(259, 4616, 1889, 2641)
salary <- salary[-rows_to_del, ]
nrow(salary)

#### Descriptive Statistics and Plots

summary(salary)

## Histogram for the Age variable
library(ggplot2)

ggplot(salary, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "maroon", color = "black", alpha = 0.7, breaks = seq(18, 65, by = 1)) +
  labs(title = "Histogram for Age",
       x = "Age",
       y = "Frequency")

## Barplot for the Gender variable
ggplot(salary, aes(x = Gender, fill = Gender)) +
  geom_bar() +
  labs(title = "Genders",
       x = "Gender",
       y = "Frequency")

## Barplot for the Position Names

# How many unique position names in the dataset?
unique_job_titles <- unique(salary$Job.Title)
sayı_unique_job_titles <- length(unique_job_titles)
print(sayı_unique_job_titles)

# Bar plot for top 10 highest frequency positions.

job_title_sayım <- table(salary$Job.Title)
top10_job_titles <- names(sort(job_title_sayım, decreasing = TRUE)[1:10])

ggplot(salary[salary$Job.Title %in% top10_job_titles, ], aes(x = Job.Title, fill = Job.Title)) +
  geom_bar() +
  labs(title = "Top 10 Position Names",
       x = "Position Name",
       y = "Frequency") +
  theme(axis.text.x = element_text(angle = 65, hjust = 1))

## Histogram for Salary
ggplot(salary, aes(x = Salary)) +
  geom_histogram(breaks = seq(0, 270000, by = 10000), fill = "chocolate", color = "black", alpha = 0.7) +
  labs(title = "Salary Histogram",
       x = "Yearly Salary",
       y = "Frequency") +  scale_y_continuous(breaks = seq(0, 600, by = 100))


#### Correlation Matrix

library(corrplot)

# Selecting the numeric variables for corr plot
sayısal_kolonlar <- sapply(salary, is.numeric)
sayısal_data <- salary[, sayısal_kolonlar]

# Corr plot itself
cor_matrix <- cor(sayısal_data)

corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.7, cl.cex = 0.8, addrect = 3)



#### MULTIPLE LINEAR REGRESSION MODEL AND a. SIGNIFICANT COEFFICIENTS, b. R2 and adj-R2 COEFFICIENTS, c. VIF VALUES

## Encoding the categorical variables

library(dplyr)

cols_to_encode <- c("Gender", "Job.Title", "Country", "Race")

# Encoding with "model.matrix" function
encoded_data <- model.matrix(~ . + 0, data = salary[, cols_to_encode])

# Combining the original dataset with encoded data
salary_encoded <- cbind(salary, as.data.frame(encoded_data))

# Deleting the categorical columns in the original dataset from the encoded data.
salary_encoded <- salary_encoded[, !colnames(salary) %in% cols_to_encode]

## Multiple Linear Regression to predict salary

# Regression
lm_model <- lm(Salary ~ ., data = salary_encoded)

lm_model

# Regression Plot
ggplot(salary_encoded, aes(x = Salary, y = lm_model$fitted.values)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Observed vs. Predicted Salary",
       x = "Observed Salary",
       y = "Predicted Salary")


## a. Significant Coefficients | b. R-squared and Adjusted R-squared | c. VIF values
summary(lm_model)

library(car)
vif(lm_model)


#### Variable Selection Method

# 1. Stepwise Regression

stepwise_model <- lm(Salary ~ ., data = salary_encoded)
stepwise_model <- step(stepwise_model)

# 2. Backward Elimination
backward_model <- lm(Salary ~ ., data = salary_encoded)
backward_model <- step(backward_model, direction = "backward")

# 3. Forward Selection
forward_model <- lm(Salary ~ ., data = salary_encoded)
forward_model <- step(forward_model, direction = "forward")

# Comparison of Models
summary(stepwise_model)
summary(backward_model)
summary(forward_model)

# Final Model: Backward Elimination

## a. Check the assumption that the errors are normally distributed graphically and with the appropriate statistical test.

# Q-Q Plot showing the distribution of residuals
qqnorm(backward_model$residuals)
qqline(backward_model$residuals, col = 2)

# Shapiro-Wilk Normality Test (shapiro-wilk test does not work because the sample size is too large)...
shapiro.test(backward_model$residuals)

#...That's why we apply the Anderson-Darling Normality Test
library(nortest)
ad.test(backward_model$residuals)

## b. Check whether the errors have constant variance using a graph and an appropriate statistical test.

# Scale-Location Plot 
ggplot(salary_encoded, aes(x = backward_model$fitted.values, y = sqrt(abs(backward_model$residuals)))) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Scale-Location Plot",
       x = "Fitted Values",
       y = "Square Root of Standardized Residuals")


# Breusch-Pagan Test of Heteroscedasticity
library(lmtest)
bptest(backward_model)

## c. Determine whether there are extreme values and effective observations with graphs and relevant values.

# Cook's Distance Plot
plot(backward_model, which = 4) 
abline(h = 4/(nrow(salary_encoded) - length(backward_model$coefficients)), col="red", lty=2)

# Standardized Residuals vs Leverage Plot
plot(backward_model, which = 5)

## d. Make an interpretation by looking at the VIF values of the final model.
vif_values <- vif(backward_model)
vif_values

## Interpret the coefficients of the final model.
summary(backward_model)

## f.	Obtain and interpret the 95% confidence intervals of the coefficients.
confint(backward_model, level = 0.95)

## g.	Find and interpret the 95% confidence interval and prediction interval for a new observation value.

yeni_gozlem <- data.frame(
  Age = 30,
  Gender = "Male",
  Education.Level = "1",
  Job.Title = "Data Scientist",
  Years.of.Experience = "3.0",
  Salary = NA,
  Country = "USA",
  Race = "Asian",
  Senior = 0
)

salary <- rbind(salary, yeni_gozlem)

salary_data <- salary_encoded$Salary
confidence_interval_yg <- t.test(salary_data, conf.level = 0.95)$conf.int
confidence_interval_yg
