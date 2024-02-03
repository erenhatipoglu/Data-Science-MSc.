
bodyfat <- read.csv("C:\\Users\\erenh\\OneDrive\\lenova\\deu data science YL\\olasılık\\final\\bodyfat.csv")

library(dplyr)
library(psych)
library(nortest) 
library(stats)  

set.seed(2023900089)

sampled_bodyfat <- bodyfat %>% sample_n(size = 150)
head(sampled_bodyfat)


### 1. Selecting 3 variables and getting the descriptive statistics

selected_variables <- sampled_bodyfat[, c("Neck", "Chest", "Biceps")]
describe(selected_variables)

### 2. Selecting the normally distributed, right and left skewed variables.

par(mfrow=c(3,5))

hist(sampled_bodyfat$Density) #normal dağılıma en yakın
hist(sampled_bodyfat$BodyFat)
hist(sampled_bodyfat$Age)
hist(sampled_bodyfat$Weight)
hist(sampled_bodyfat$Height) #soldan çarpık
hist(sampled_bodyfat$Neck)
hist(sampled_bodyfat$Chest)
hist(sampled_bodyfat$Abdomen)
hist(sampled_bodyfat$Hip) 
hist(sampled_bodyfat$Thigh)
hist(sampled_bodyfat$Knee)
hist(sampled_bodyfat$Ankle) #sağdan çarpık
hist(sampled_bodyfat$Biceps) 
hist(sampled_bodyfat$Forearm) 
hist(sampled_bodyfat$Wrist)

par(mfrow=c(1,1))

describe(sampled_bodyfat)

## Goodness-of-Fit tests
# Cramer-von-Mises
cvm.test(sampled_bodyfat$Density)
cvm.test(sampled_bodyfat$Height)
cvm.test(sampled_bodyfat$Ankle)

# Anderson-Darling
ad.test(sampled_bodyfat$Density)
ad.test(sampled_bodyfat$Height)
ad.test(sampled_bodyfat$Ankle)

# Kolmogorov-Smirnov
ks.test(sampled_bodyfat$Density, "pnorm")
ks.test(sampled_bodyfat$Height, "pnorm")
ks.test(sampled_bodyfat$Ankle, "pnorm")

# Shapiro-Wilk
shapiro.test(sampled_bodyfat$Density)
shapiro.test(sampled_bodyfat$Height)
shapiro.test(sampled_bodyfat$Ankle)


# Transforming the right skewed variable (Ankle)

ankle_karekok <- sqrt(sampled_bodyfat$Ankle)
ankle_kupkok <- (sampled_bodyfat$Ankle)^(1/3)
ankle_log <- log10(sampled_bodyfat$Ankle)
ankle_eksibir <- 1 / sampled_bodyfat$Ankle

# cvm test for transformed ankle
cvm.test(ankle_karekok)
cvm.test(ankle_kupkok)
cvm.test(ankle_log)
cvm.test(ankle_eksibir)

# AD test for transformed ankle
ad.test(ankle_karekok)
ad.test(ankle_kupkok)
ad.test(ankle_log)
ad.test(ankle_eksibir)

# KS test for transformed ankle
ks.test(ankle_karekok, "pnorm")
ks.test(ankle_kupkok, "pnorm")
ks.test(ankle_log, "pnorm")
ks.test(ankle_eksibir, "pnorm")

# Shapiro-Wilk test for transformed ankle
shapiro.test(ankle_karekok)
shapiro.test(ankle_kupkok)
shapiro.test(ankle_log)
shapiro.test(ankle_eksibir)



# Transforming the left skewed variable (Height)

height_kare <- (sampled_bodyfat$Height)^2
height_kup <- (sampled_bodyfat$Height)^3

# cvm test for transformed ankle
cvm.test(height_kare)
cvm.test(height_kup)

# ad test for transformed ankle
ad.test(height_kare)
ad.test(height_kup)

# KS test for transformed ankle
ks.test(height_kare, "pnorm")
ks.test(height_kup, "pnorm")

# Shapiro-Wilk test for transformed ankle
shapiro.test(height_kare)
shapiro.test(height_kup)


### 3. 

# Function for calculating the sample means
demonstrate_CLT <- function(variable, n, num_samples = 1000) {
  sample_means <- numeric(num_samples)
  
  for (i in 1:num_samples) {
    sample <- sample(variable, size = n, replace = TRUE)
    sample_means[i] <- mean(sample)
  }
  
  return(sample_means)
}


dnormal <- sampled_bodyfat$Density
dsaga_carpik <- sampled_bodyfat$Ankle
dsola_carpik <- sampled_bodyfat$Height

# Sample size
sample_sizes <- c(5, 10, 30, 50)

# Proving the Central Limit Theorem for every variable and sample size
results <- list()

for (size in sample_sizes) {
  results[[paste("Normal", size)]] <- demonstrate_CLT(dnormal, size)
  results[[paste("Saga_Carpik", size)]] <- demonstrate_CLT(dsaga_carpik, size)
  results[[paste("Sola_Carpik", size)]] <- demonstrate_CLT(dsola_carpik, size)
}

# Plot
par(mfrow=c(3, 4))

# normally distributed ones
for (result in grep("Normal", names(results), value = TRUE)) {
  hist(results[[result]], main = result, xlab = "Örneklem Ortalaması", breaks = 30, col = "grey")
}

# right skewed ones
for (result in grep("Saga_Carpik", names(results), value = TRUE)) {
  hist(results[[result]], main = result, xlab = "Örneklem Ortalaması", breaks = 30, col = "cornsilk")
}

# left skewed ones
for (result in grep("Sola_Carpik", names(results), value = TRUE)) {
  hist(results[[result]], main = result, xlab = "Örneklem Ortalaması", breaks = 30, col = "lightblue")
}

### 4.

# Random sample from Density variable with a size of 30
set.seed(2023900089)
sample_size <- 30
sample_density <- sample(sampled_bodyfat$Density, size = sample_size, replace = TRUE)

# Mean and standart deviation of the sample
sample_mean <- mean(sample_density)
sample_sd <- sd(sample_density)

# 99% confidence interval for the population mean
alpha_mean <- 0.01
z_critical_mean <- qnorm(1 - alpha_mean / 2) # Z distribution

mean_error_margin <- z_critical_mean * (sample_sd / sqrt(sample_size))
mean_confidence_interval <- c(sample_mean - mean_error_margin, sample_mean + mean_error_margin)

# 90% confidence interval for the population variance
alpha_variance <- 0.10
chi_critical_lower <- qchisq(alpha_variance / 2, df = sample_size - 1)
chi_critical_upper <- qchisq(1 - alpha_variance / 2, df = sample_size - 1)

variance_error_margin_lower <- ((sample_size - 1) * sample_sd^2) / chi_critical_upper
variance_error_margin_upper <- ((sample_size - 1) * sample_sd^2) / chi_critical_lower
variance_confidence_interval <- c(variance_error_margin_lower, variance_error_margin_upper)

# Printing out the 99% confidence interval for the mean
mean_confidence_interval

# Printing out the 90% confidence interval for the population variance
variance_confidence_interval

