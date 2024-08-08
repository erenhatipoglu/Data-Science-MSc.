earth <- read.csv("D:/deu data science YL/2.donem/kesifsel/donem_odevi/quake.csv")

library(tidyverse)
library(psych)
library(dplyr)
library(corrplot)
library(ggcorrplot)
library(deldir)
library(VIM)
library(tseries)
library(factoextra)
library(cluster)
library(leaflet)
library(mice)


head(earth)
str(earth)
dim(earth)
summary(earth)
anyNA(earth)


# Kay??p verilerin oldu??u kolonlar??n belirlenmesi
columns_with_na <- earth %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(cols = everything(), names_to = "column", values_to = "na_count") %>%
  filter(na_count > 0) %>%
  pull(column)

# Say??sal ve kategorik de??i??kenlerin ayr??lmas??
numeric_vars <- earth %>% select(where(is.numeric))
categorical_vars <- earth %>% select(where(is.character))

# Say??sal verilerin KNN kullanarak impute edilmesi
numeric_vars_imputed <- kNN(numeric_vars, k = sqrt(nrow(numeric_vars)))

# Kategorik verilerin mod kullan??larak impute edilmesi
impute_mode <- function(x) {
  x[is.na(x)] <- as.character(getmode(x))
  return(x)
}

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

categorical_vars_imputed <- categorical_vars %>% 
  mutate(across(everything(), impute_mode))

# Verisetinin atanm???? verilerle g??ncellenmesi
earth[names(numeric_vars)] <- numeric_vars_imputed[names(numeric_vars)]
earth[names(categorical_vars)] <- categorical_vars_imputed[names(categorical_vars)]

earth %>% anyNA



# Numerik de??i??kenler i??in ??zet
summary_stats <- earth %>%
  select_if(is.numeric) %>%
  describe()

summary_stats


# Say??sal de??i??kenler i??in histogram
earth %>%
  select_if(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free") +
  labs(title = "Say??sal De??i??kenlerin Histogramlar??")


# Korelasyon matrisi
correlation_matrix <- earth %>%
  select_if(is.numeric) %>%
  cor()

ggcorrplot(correlation_matrix, 
           method = "circle",
           hc.order = TRUE,
           type = "lower",
           lab = TRUE,         
           title = "Say??sal De??i??kenlerin Korelasyon Matrisi")


# ??zet istatistiklerin group by ile ????kar??lmas??
earth %>%
  group_by(type) %>%
  summarise(
    count = n(),
    mean_mag = mean(mag),
    median_mag = median(mag),
    sd_depth = sd(depth)
  )


# Boxplot
earth %>%
  ggplot(aes(x = type, y = mag)) +
  geom_boxplot() +
  labs(title = "Boxplot of Magnitude by Earthquake Type")


# 4- tidyverse fonksiyonlari ile arastirma probleminize yonelik gelismis grafikler olusturun.

# Time series plot

# Tidyverse kullanarak 'time'?? date format??na d??n????t??rme
earth <- earth %>%
  mutate(time = ymd_hms(time))

earth_subset <- earth %>%
  filter(date(time) >= as.Date("2024-01-01") & date(time) <= as.Date("2024-03-31"))

earth_ts <- ts(earth_subset$mag, start = c(2024, 1), end = c(2024, 90), frequency = 365)

# ggplot2 kullanarak g??rselle??tirme
ggplot(data.frame(Time = as.Date(365 * (1:90)/365, origin = "2024-01-01"), Magnitude = as.vector(earth_ts)), aes(x = Time, y = Magnitude)) +
  geom_line() +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %d") +
  labs(title = "Earthquake Magnitude (Jan-Mar 2024)", x = "Date", y = "Magnitude")



# Deprem lokasyonlar??n??n harita ??zerinde g??rselle??tirilmesi
earth %>%
  ggplot(aes(x = longitude, y = latitude, color = mag)) +
  geom_point() +
  labs(title = "Spatial Distribution of Earthquakes by Magnitude", x = "Longitude", y = "Latitude")




# Clustering i??in magnitude ve depth de??i??kenlerinin se??ilmesi
clustering_vars <- earth %>%
  select(mag, depth)

# Clustering i??in subset al??nmas??
set.seed(123)
subset_earth <- clustering_vars %>%
  sample_n(1000)

# Optimum cluster say??s??na karar verilmesi
# Elbow Method
fviz_nbclust(subset_earth, kmeans, method = "wss") +
  geom_vline(xintercept = 10, linetype = 2, color = "red") +
  labs(title = "Elbow Method for Choosing k")

# Silhouette Analysis
fviz_nbclust(subset_earth, kmeans, method = "silhouette") +
  labs(title = "Silhouette Analysis for Choosing k")

# Gap Statistic
fviz_nbclust(subset_earth, kmeans, nstart = 25, method = "gap_stat", nboot = 100)

# 7 adet k??me say??s??na karar verildi.
set.seed(123)
km_res <- kmeans(subset_earth, centers = 7, nstart = 25)

# K??me etiketleri
subset_earth <- subset_earth %>%
  mutate(cluster = as.factor(km_res$cluster))

summary(subset_earth)

# Cluster grafi??i
ggplot(subset_earth, aes(x = depth, y = mag, color = cluster)) +
  geom_point(alpha = 0.5, size = 2) +
  labs(title = "K-Means Clustering (k = 7)",
       x = "Depth",
       y = "Magnitude") +
  theme_minimal()

# Her bir k??me i??in ortalama derinlik ve b??y??kl??k
cluster_summary <- subset_earth %>%
  group_by(cluster) %>%
  summarise(mean_depth = mean(depth), mean_mag = mean(mag))

print(cluster_summary)



