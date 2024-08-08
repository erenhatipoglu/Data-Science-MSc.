# Load necessary libraries
library(tidyverse)
library(TTR)
library(forecast)
library(tseries)
library(lmtest)
library(keras)
library(tensorflow)
library(tsibble)
library(timetk)
library(reticulate)
library(e1071)
library(strucchange)
library(caret)

cpi <- read.csv("D:/deu data science YL/2.donem/zaman serileri/final/US CPI.csv")

cpi$Yearmon <- as.Date(cpi$Yearmon)

cpi_ts <- ts(cpi$CPI, start=c(1913, 1), frequency=12)

plot(cpi_ts, main = "US CPI Time Series")

# Decomposition
cpi_decompose <- decompose(cpi_ts, type="additive")
plot(cpi_decompose)

# Box-Cox transformation
lambda <- BoxCox.lambda(cpi_ts)
cpi_boxcox <- BoxCox(cpi_ts, lambda)
plot(cpi_boxcox, main = "Box-Cox Transformed CPI")

diff1 <- diff(cpi_boxcox)
plot(diff1, main = "First Differencing of Box-Cox Transformed CPI")

diff2 <- diff(diff1)
plot(diff2, main = "Second Differencing of Box-Cox Transformed CPI")


acf(diff2, lag.max=50, main="ACF of Differenced Series")
pacf(diff2, lag.max=50, main="PACF of Differenced Series")


#ARIMA(1,2,1)
model_121 <- arima(cpi_ts, order=c(1, 2, 1))
aic_121 <- AIC(model_121)
cat("AIC of ARIMA(1,2,1):", aic_121, "\n")

#ARIMA(2,2,1)
model_221 <- arima(cpi_ts, order=c(2, 2, 1))
aic_221 <- AIC(model_221)
cat("AIC of ARIMA(2,2,1):", aic_221, "\n")

#ARIMA(1,2,2)
model_122 <- arima(cpi_ts, order=c(1, 2, 2))
aic_122 <- AIC(model_122)
cat("AIC of ARIMA(1,2,2):", aic_122, "\n")

#AIC
aic_values <- c("ARIMA(1,2,1)"=aic_121, "ARIMA(2,2,1)"=aic_221, "ARIMA(1,2,2)"=aic_122)
aic_values

# lowest AIC
best_model <- names(which.min(aic_values))
cat("The model with the lowest AIC is:", best_model, "\n")

model_221 %>% coeftest
model_221 %>% tsdiag

Box.test(model_221$residuals, lag=36)
Box.test(model_221$residuals, lag=24)
Box.test(model_221$residuals, lag=12)
Box.test(model_221$residuals, lag=6)

model_221 %>% checkresiduals

fore <- forecast(model_221, h=12)
fore %>% plot
fore %>% summary


oto <- auto.arima(cpi_ts)
oto
oto %>% summary
oto %>% tsdiag

### LSTM

# Function to scale data
scaler <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Function to inverse scale data
inverse_scaler <- function(x, original_data) {
  x * (max(original_data) - min(original_data)) + min(original_data)
}

# Load and scale the CPI data
scaled_cpi <- scaler(cpi$CPI)

# Create sequences for training
create_sequences <- function(data, window_size) {
  sequences <- list()
  targets <- list()
  for (i in seq(window_size, length(data) - 1)) {
    sequences[[i - window_size + 1]] <- data[(i - window_size + 1):i]
    targets[[i - window_size + 1]] <- data[i + 1]
  }
  list(sequences = do.call(rbind, sequences), targets = unlist(targets))
}

# Define window size and create sequences
window_size <- 12
data_sequences <- create_sequences(scaled_cpi, window_size)

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data_sequences$targets, p = 0.8, list = FALSE)
train_set <- data_sequences$sequences[train_index, , drop = FALSE]
test_set <- data_sequences$sequences[-train_index, , drop = FALSE]
train_target <- data_sequences$targets[train_index]
test_target <- data_sequences$targets[-train_index]

# Reshape data for LSTM (samples, timesteps, features)
train_set <- array(train_set, dim = c(nrow(train_set), window_size, 1))
test_set <- array(test_set, dim = c(nrow(test_set), window_size, 1))

# Build LSTM model using keras
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, activation = "relu", input_shape = c(window_size, 1)) %>%
  layer_lstm(units = 50, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 0.02))

# Train the model
history <- model %>% fit(train_set, train_target, epochs = 15, validation_data = list(test_set, test_target))

# Predict future values
future_data <- tail(scaled_cpi, window_size)
future_data <- array(future_data, dim = c(1, window_size, 1))
predicted_data <- model %>% predict(future_data)

# Inverse scale predictions
predicted_cpi <- inverse_scaler(predicted_data, cpi$CPI)

# Print predicted values for the next month
cat("Predicted CPI for next month:\n")
print(predicted_cpi)

# Calculate MAPE
actual <- cpi$CPI[(length(cpi$CPI) - window_size + 1):length(cpi$CPI)]
mape <- mean(abs((actual - predicted_cpi) / actual))
cat("MAPE for the predictions is:", mape, "\n")

# Plot LSTM forecast
plot(fore, main = "Forecast using LSTM", xlab = "Time", ylab = "CPI", xlim = c(1913, 2025))
legend("topleft", legend = c("Actual CPI", "LSTM Forecast"), col = c("black", "red"), lty = 1, cex = 0.8)


### SVM

# Train SVM model
model_svm <- svm(train_target ~ ., data = data.frame(train_set), kernel = "radial")

# Predict future values
future_data <- tail(scaled_cpi, window_size)
predicted_data <- predict(model_svm, newdata = data.frame(t(tail(future_data, window_size))))

# Inverse scale predictions
predicted_cpi <- inverse_scaler(predicted_data, cpi$CPI)

# Print predicted values for the next month
cat("Predicted CPI for next month:\n")
print(predicted_cpi)

# Calculate MAPE
actual <- cpi$CPI[(length(cpi$CPI) - window_size + 1):length(cpi$CPI)]
mape <- mean(abs((actual - predicted_cpi) / actual))
cat("MAPE for the predictions is:", mape, "\n")


# Plot SVM forecast
plot(cpi_ts, main = "Forecast using SVM", xlab = "Time", ylab = "CPI", xlim = c(1913, 2025))
legend("topleft", legend = c("Actual CPI", "SVM Forecast"), col = c("black", "blue"), lty = 1, cex = 0.8)


