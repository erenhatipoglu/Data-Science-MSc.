library(readxl)
library(corrplot)
library(tidyverse)
library(factoextra)
library(cluster)
library(fpc)
library(fossil)
library(NbClust)
library(mclust)
library(dbscan)
library(clustertend)
library(clValid)
library(stats)
library(hopkins)

# Read the data
data <- read_excel("BreastTissue.xlsx")

# Check the structure, dimensions, and head of the data
str(data)
dim(data)
head(data)

# Remove the first two columns
numeric_data <- data[,-c(1,2)]

# Check the structure of the numeric data
str(numeric_data)

# Convert data to long format
long_data <- pivot_longer(numeric_data, cols = everything(), names_to = "variable", values_to = "value")

# Create boxplots for original data
ggplot(long_data, aes(x = variable, y = value)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplots of Numeric Variables in BreastTissue Dataset",
       x = "Variable",
       y = "Value")

# Scale the numeric data
scaled_data <- scale(numeric_data)

# Convert the scaled matrix back to a data frame
scaled_data <- as.data.frame(scaled_data)

# Convert scaled data to long format
long_scaled_data <- pivot_longer(scaled_data, cols = everything(), names_to = "variable", values_to = "value")

# Create boxplots for scaled data
ggplot(long_scaled_data, aes(x = variable, y = value)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplots of Scaled Numeric Variables in BreastTissue Dataset",
       x = "Variable",
       y = "Value")




corr <- cor((numeric_data), method = "pearson")
corr
corrplot.mixed(corr, lower="pie",upper="number")



# Principal Component Analysis
data.pca <- prcomp(numeric_data, center = TRUE, scale. = TRUE)
summary(data.pca)
(data.pca$sdev)^2 # Eigenvalues

fviz_eig(data.pca)

data.pca$rotation # Eigenvectors
data.pca$x[,1:2] 
cor(data.pca$x[,1],data.pca$x[,2])
data_pca <- data.pca$x[,1:2] 

# Results for Variables
res.var <- get_pca_var(data.pca)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 

fviz_pca_var(data.pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE # Avoid text overlapping
)

fviz_pca_var(data.pca, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE # Avoid text overlapping
)

# Results for individuals
res.ind <- get_pca_ind(data.pca)
res.ind$coord          # Coordinates
res.ind$contrib        # Contributions to the PCs
res.ind$cos2           # Quality of representation

fviz_pca_ind(data.pca,
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# Distance calculations
dist_euc <- get_dist(scaled_data, stand = TRUE, method = "euclidean")
fviz_dist(dist_euc)

dist_man <- get_dist(scaled_data, stand = TRUE, method = "manhattan")
fviz_dist(dist_man)

# Hopkins statistic for clustering tendency
set.seed(123)
h_data <- hopkins(data_pca, nrow(data_pca) - 1)
h_data

# Number of clusters using NbClust
nb <- NbClust(data_pca, distance = "euclidean", min.nc = 2, max.nc = 9, method = "kmeans")

fviz_nbclust(data_pca, kmeans, nstart = 25, iter.max = 200, method = "wss") +
  labs(subtitle = "Elbow method")

fviz_nbclust(data_pca, kmeans, nstart = 25, iter.max = 200, method = "silhouette") +
  labs(subtitle = "Silhouette method")

fviz_nbclust(data_pca, kmeans, nstart = 25, method = "gap_stat", nboot = 500) +
  labs(subtitle = "Gap statistic method")

# k-means clustering
set.seed(123)
km_data <- kmeans(data_pca, 4, nstart = 25) 
print(km_data)

fviz_cluster(km_data, data = data_pca,
             ellipse.type = "convex", 
             star.plot = TRUE, 
             repel = TRUE, 
             ggtheme = theme_minimal()
)

summary(data[,-c(1,2)])
aggregate(data[,-c(1,2)], by = list(cluster = km_data$cluster), mean) 

# k-medoids clustering
fviz_nbclust(data_pca, pam, method= "silhouette")

set.seed(123)
pam_data_4 <- pam(data_pca, 4)
print(pam_data_4)

fviz_cluster(pam_data_4,
             ellipse.type = "convex", 
             repel = TRUE, 
             ggtheme = theme_classic()
)

# Hierarchical clustering
hc_e <- hclust(d = dist_euc, method = "ward.D2")
fviz_dend(hc_e, cex = 0.5)

grup <- cutree(hc_e, k = 4)
fviz_dend(hc_e, k = 4, cex = 0.5, color_labels_by_k = TRUE, rect = TRUE)

fviz_cluster(list(data = data_pca, cluster = grup),
             palette = c("#2E9FDF", "#00FF00", "#E7B800", "#FA4E07", "Maroon"),
             ellipse.type = "convex", 
             repel = TRUE, 
             show.clust.cent = FALSE, 
             ggtheme = theme_minimal()
)

summary(data[,-c(1,2)])
aggregate(data[,-c(1,2)], by = list(cluster = grup), mean)

# Model-based clustering
mc <- Mclust(data_pca)
summary(mc)
mc$z

fviz_mclust(mc, "BIC", palette = "jco")
fviz_mclust(mc, "classification", geom = "point", pointsize = 1.5, palette = "jco")
fviz_mclust(mc, "uncertainty", palette = "jco")

summary(data[,-c(1,2)])
aggregate(data[,-c(1,2)], by = list(cluster = mc$classification), mean)

# Density-based clustering
set.seed(123)
dbscan::kNNdistplot(data_pca, k = sqrt(nrow(data_pca)))
db <- fpc::dbscan(data_pca, eps = 1.2, MinPts = 3)
print(db)

fviz_cluster(db, data = data_pca, stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE,
             geom = "point", palette = "jco", ggtheme = theme_classic()
)

summary(data[,-c(1,2)])
aggregate(data[,-c(1,2)], by = list(cluster = db$cluster), mean)




# Define a function to compute silhouette score
compute_silhouette <- function(cluster_result, data) {
  if (is(cluster_result, "kmeans") | is(cluster_result, "pam")) {
    cluster_labels <- cluster_result$cluster
  } else if (is(cluster_result, "hclust")) {
    cluster_labels <- cutree(cluster_result, k = 4)  # Adjust 'k' as necessary
  } else if (is(cluster_result, "Mclust")) {
    cluster_labels <- cluster_result$classification
  } else if (is(cluster_result, "dbscan")) {
    cluster_labels <- cluster_result$cluster
  } else {
    stop("Unknown clustering result type")
  }
  
  silhouette_score <- silhouette(cluster_labels, dist(data))
  mean(silhouette_score[, 3])
}

# Calculate silhouette scores for each clustering method
sil_kmeans <- compute_silhouette(km_data, data_pca)
sil_pam <- compute_silhouette(pam_data_4, data_pca)
sil_hclust <- compute_silhouette(hc_e, data_pca)
sil_mclust <- compute_silhouette(mc, data_pca)
sil_dbscan <- compute_silhouette(db, data_pca)

# Print silhouette scores
cat("Silhouette Scores:\n")
cat("K-means: ", sil_kmeans, "\n")
cat("K-medoids: ", sil_pam, "\n")
cat("Hierarchical: ", sil_hclust, "\n")
cat("Model-based: ", sil_mclust, "\n")
cat("DBSCAN: ", sil_dbscan, "\n")


summary(data[,-c(1,2)])
aggregate(data[,-c(1,2)], by = list(cluster = km_data$cluster), mean)

summary(data[,-c(1,2)])
aggregate(data[,-c(1,2)], by = list(cluster = km_data$cluster), sd) 

