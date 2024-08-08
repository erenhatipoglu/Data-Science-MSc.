# Load the required libraries
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

# Read the dataset
earth <- read.csv("D:/deu data science YL/2.donem/kesifsel/donem_odevi/quake.csv")

# Display the first few rows of the dataset
head(earth)
# Display the structure of the dataset
str(earth)
# Display the dimensions of the dataset
dim(earth)
# Display summary statistics of the dataset
summary(earth)
# Check for any missing values in the dataset
anyNA(earth)

# Identify columns with missing values
columns_with_na <- earth %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(cols = everything(), names_to = "column", values_to = "na_count") %>%
  filter(na_count > 0) %>%
  pull(column)

# Separate numeric and categorical variables
numeric_vars <- earth %>% select(where(is.numeric))
categorical_vars <- earth %>% select(where(is.character))

# Impute missing values in numeric variables using KNN
numeric_vars_imputed <- kNN(numeric_vars, k = sqrt(nrow(numeric_vars)))

# Impute missing values in categorical variables using mode
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

# Update the dataset with imputed values
earth[names(numeric_vars)] <- numeric_vars_imputed[names(numeric_vars)]
earth[names(categorical_vars)] <- categorical_vars_imputed[names(categorical_vars)]

# Check if there are any missing values remaining
earth %>% anyNA

# Summary statistics for numeric variables
summary_stats <- earth %>%
  select_if(is.numeric) %>%
  describe()

summary_stats

# Histogram for numeric variables
earth %>%
  select_if(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free") +
  labs(title = "Histograms of Numeric Variables")

# Correlation matrix
correlation_matrix <- earth %>%
  select_if(is.numeric) %>%
  cor()

ggcorrplot(correlation_matrix, 
           method = "circle",
           hc.order = TRUE,
           type = "lower",
           lab = TRUE,         
           title = "Correlation Matrix of Numeric Variables")

# Summary statistics grouped by 'type'
earth %>%
  group_by(type) %>%
  summarise(
    count = n(),
    mean_mag = mean(mag),
    median_mag = median(mag),
    sd_depth = sd(depth)
  )

# Boxplot of magnitude by earthquake type
earth %>%
  ggplot(aes(x = type, y = mag)) +
  geom_boxplot() +
  labs(title = "Boxplot of Magnitude by Earthquake Type")

# Advanced graphics for the research problem using tidyverse functions

# Convert 'time' to date format using tidyverse
earth <- earth %>%
  mutate(time = ymd_hms(time))

# Filter data for a specific date range
earth_subset <- earth %>%
  filter(date(time) >= as.Date("2024-01-01") & date(time) <= as.Date("2024-03-31"))

# Create a time series object for magnitude
earth_ts <- ts(earth_subset$mag, start = c(2024, 1), end = c(2024, 90), frequency = 365)

# Time series plot using ggplot2
ggplot(data.frame(Time = as.Date(365 * (1:90)/365, origin = "2024-01-01"), Magnitude = as.vector(earth_ts)), aes(x = Time, y = Magnitude)) +
  geom_line() +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %d") +
  labs(title = "Earthquake Magnitude (Jan-Mar 2024)", x = "Date", y = "Magnitude")

# Plot the spatial distribution of earthquakes by magnitude
earth %>%
  ggplot(aes(x = longitude, y = latitude, color = mag)) +
  geom_point() +
  labs(title = "Spatial Distribution of Earthquakes by Magnitude", x = "Longitude", y = "Latitude")

# Select variables for clustering (magnitude and depth)
clustering_vars <- earth %>%
  select(mag, depth)

# Sample a subset of the data for clustering
set.seed(123)
subset_earth <- clustering_vars %>%
  sample_n(1000)

# Determine the optimal number of clusters

# Elbow Method
fviz_nbclust(subset_earth, kmeans, method = "wss") +
  geom_vline(xintercept = 10, linetype = 2, color = "red") +
  labs(title = "Elbow Method for Choosing k")

# Silhouette Analysis
fviz_nbclust(subset_earth, kmeans, method = "silhouette") +
  labs(title = "Silhouette Analysis for Choosing k")

# Gap Statistic
fviz_nbclust(subset_earth, kmeans, nstart = 25, method = "gap_stat", nboot = 100)

# Decided to use 7 clusters
set.seed(123)
km_res <- kmeans(subset_earth, centers = 7, nstart = 25)

# Add cluster labels to the subset
subset_earth <- subset_earth %>%
  mutate(cluster = as.factor(km_res$cluster))

# Summary of clustering results
summary(subset_earth)

# Plot clusters
ggplot(subset_earth, aes(x = depth, y = mag, color = cluster)) +
  geom_point(alpha = 0.5, size = 2) +
  labs(title = "K-Means Clustering (k = 7)",
       x = "Depth",
       y = "Magnitude") +
  theme_minimal()

# Summary of mean depth and magnitude for each cluster
cluster_summary <- subset_earth %>%
  group_by(cluster) %>%
  summarise(mean_depth = mean(depth), mean_mag = mean(mag))

print(cluster_summary)
