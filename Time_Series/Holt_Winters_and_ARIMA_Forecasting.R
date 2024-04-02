library(TTR)
library(forecast)
library(tseries)
library(lmtest)

comp <- scan("https://robjhyndman.com/tsdldata/data/computer.dat",skip = 0)

comp_ts <- ts(comp, start = c(2000,1), frequency = 12)

plot(comp_ts)

#There is no seasonality, random shocks and trends may exist.

### Decomposition and removal of the seasonality

comp_decompose <- decompose(comp_ts, type="additive")
plot(comp_decompose)
new_comp <- comp_ts - comp_decompose$seasonal
plot(new_comp)


#The graph obtained when we separate it into its components and remove seasonality.

#Smoothing Moving Average

comp_sma <- SMA(comp_ts, n=10)
plot(comp_sma)

### Exponential Smoothing

comp_holt <- HoltWinters(comp_ts, beta=F, gamma = F)
comp_holt
comp_holt$SSE
plot(comp_holt)

comp_holt2 <- HoltWinters(comp_ts, beta=T, gamma = F) #Trend bileşeni var olan daha iyi çıkıyor
comp_holt2
comp_holt2$SSE
plot(comp_holt2)

comp_holt_fore <- forecast(comp_holt2, h=5)
comp_holt_fore
plot(comp_holt_fore)


#We see that the Holt-Winters exponential smoothing method gives better results when the trend (beta = TRUE) is added to the function. 
#When Beta is false, the Error Sum of Squares is 2.6 times higher than the other one. This shows that there is a difference of 2.6 times 
#between what actually happened and what was predicted.

#Alpha values are high in both, which means more weight is given to the latest observations.
#When the prediction is made, the graph with confidence intervals can be observed above.

### Differencing


adf.test(comp_ts)
comp_diff <- diff(comp_ts, diff=1)
plot(comp_diff)
comp_diff2 <- diff(comp_ts, diff=2) #(d değeri)
plot(comp_diff2)


#Since the original time series was not stationary according to the ACF test result, 
#the 1st and 2nd order differences were taken and made stationary.

#When the 2nd order difference is taken, the series seems to become stationary, 
#but when the 1st order difference is taken, it does not become stationary.

### Augmented Dickey-Fuller Test


adf.test(comp_diff)
adf.test(comp_diff2) #Only diff2 is stationary.

#When we take the second-order difference and apply the ACF test, we reject the null hypothesis and say that the series becomes stationary.

### ACF ve Partial-ACF Plots 

plot(comp_ts)
acf(comp_diff2, lag.max=50) #ar (p) 
pacf(comp_diff2, lag.max=50) #ma (q) 

#Cut-off is observed in both graphs, it can be said that there is no significant autocorrelation.

### Temporary Models

comp_arima1 <- arima(comp_ts, order=c(0,2,2))
comp_arima1
coeftest(comp_arima1)
tsdiag(comp_arima1)

comp_arima2 <- arima(comp_ts, order=c(2,2,2))
comp_arima2
coeftest(comp_arima2)
tsdiag(comp_arima2)

comp_arima3 <- auto.arima(comp_ts) #auto.arima
comp_arima3
coeftest(comp_arima3)
tsdiag(comp_arima3)

comp_arima4 <- arima(comp_ts, order=c(2,2,0)) #Seçilen model
comp_arima4
coeftest(comp_arima4)
tsdiag(comp_arima4)

comp_arima5 <- arima(comp_ts, order=c(2,2,1)) 
comp_arima5
coeftest(comp_arima5)
tsdiag(comp_arima5)


#Various ARIMA models were created by looking at the lags. Various models were tried, but the model with the lowest AIC value was:

#- The p value was given a value of 2, taking into account the first two lags.
#- A difference of 2 degrees was chosen for the d value.
#- The value of q was given as 0.

#When we look at Coeftests, we see that SMA(1) is not significant in autoarima. When we look at the coefficient test of the selected model, we see that AR(1) and AR(2) are also significant.

### Box-Ljung ve Shapiro test

Box.test(comp_arima4$residuals, lag = 6, type = "Ljung-Box")
Box.test(comp_arima4$residuals, lag = 12, type = "Ljung-Box")
Box.test(comp_arima4$residuals, lag = 24, type = "Ljung-Box")
checkresiduals(comp_arima4)

#According to the Box-Ljung test and checkresiduals function, there is no autocorrelation between the residuals.
#According to the Shapiro-Wilk test, the residuals are normally distributed.

### Forecasting

arima_fore <- forecast(comp_arima4, h=5)
plot(arima_fore)
auto_arima_fore <- forecast(comp_arima3, h=5)
plot(auto_arima_fore)
predict(arima_fore)
accuracy(arima_fore)

#The ARIMA prediction chart made with the selected model can be seen above.

### Accuracy

accuracy(arima_fore)
accuracy(auto.arima(comp_ts))

#The error values and MAPE value of the prediction made with Auto Arima are lower than the selected model.
#Yet Auto Arima chooses the non-stationary state of the time series, therefore our model is more reliable.
