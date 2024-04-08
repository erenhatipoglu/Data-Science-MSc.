# Data: https://www.prosperity.com/rankings

library(cluster)
library(corrplot)
library(psych)
library(factoextra)
library(NbClust)

### Data Preprocessing
 
pros <- read.csv("pros.csv", row.names = 1)

pros$Safety...Security <- as.numeric(trimws(pros$Safety...Security))
pros$Personal.Freedom <- as.numeric(trimws(pros$Personal.Freedom))
pros$Governance <- as.numeric(trimws(pros$Governance))
pros$Social.Capital <- as.numeric(trimws(pros$Social.Capital))
pros$Investment.Environment <- as.numeric(trimws(pros$Investment.Environment))
pros$Education <- as.numeric(trimws(pros$Education))

pros <- na.omit(pros)
pros <- pros[,-c(1:2)]
 
# After reading the dataset, string type variables were converted into numbers, 
# 8 lines containing NA values that emerged due to forcing were deleted. 
# The first two variables of the original dataset, "Rank" and "Average.Score", were removed.

### 1. Descriptive Statistics

dim(pros)
str(pros)
 
# When we look at the descriptive statistics, 
# we see that the data consists of 159 observations and 12 lines and all variables are numerical type.
 
describe(pros)
 
#When examined with the Describe function, there does not appear to be a serious skew problem in the variables,
#and it is not possible to clearly observe whether there are extreme values without examining them with a boxplot.
 
scaled_pros <- scale(pros)
boxplot(scaled_pros, horizontal = TRUE)
 
# When the data is standardized and a box plot is created,
# we see that there are not so many extreme values that would affect the analysis, there are only 6 extreme values.
 
corr <- cor(pros, method = "pearson")
corrplot.mixed(corr, lower="pie",upper="number")
 
# The correlation graph shows us that there is a high positive correlation between the variables in almost all of them. 
# This allows us to comment that it is normal for the characteristics of countries, 
# for example, as the standard of living increases, the quality of education also increases.

### 2. PCA

scaled_pros_pca <- prcomp(scaled_pros)
summary(scaled_pros_pca)
eigenvalues <- scaled_pros_pca$sdev^2
eigenvalues
prop.var <- eigenvalues / sum(eigenvalues)
cum.prop.var <- cumsum(prop.var)
plot(prop.var , xlab=" Principal Component ", ylab=" Proportion of
Variance Explained ", ylim=c(0,1) ,type='b')
lines(cum.prop.var, type='b', col="red")
fviz_eig(scaled_pros_pca, addlabels = TRUE)
 

# In the first principal component, an explanatory rate of 74% was obtained with an eigenvalue of 8.93. 
# Although the eigenvalue of the second principal component is greater than 1, 
# it was decided to perform cluster analysis with a single component because its explanatory value was much lower than the first component.
 
# Results for Variables
res.var <- get_pca_var(scaled_pros_pca)

corrplot(res.var$cos2, is.corr = FALSE) # Quality of representation

# When we look at the PCA plot with Corrplot, we can see that almost all variables contribute to the first PCA in high amounts.
 
# Contributions of variables to PC1
fviz_contrib(scaled_pros_pca, choice = "var", axes = 1, top = 13)

# In the bar chart showing the contribution of each variable to the first principal component,
# we can say that almost all variables have a high contribution,
# but the last 4 variables do not have a significant contribution since they are below the dashed red line.
 
# Graph of variables
fviz_pca_var(scaled_pros_pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
 
# In order to draw a biplot and make an interpretation, 2 basic components were necessarily used, 
# and when we look at it, we see a positive correlation between all variables, 
# but we also see a strong positive correlation between variables such as Living Conditions and Education,
# and when interpreted by looking at the colors, if 2 components are used,
# the "Social Capital" variable is the one that contributes the least. We can observe that it is.
 
pca_pros_final <- scaled_pros_pca$x[,1]
 

### 3. Uzaklık Değerlerinin Heatmap Üzerinden Yorumlanması

dist_eucl <- dist(pca_pros_final, method="euclidean")
fviz_dist(dist_eucl)
 

# Since 1 principal component was used, Euclidean distance and Manhattan distance gave the same graphs,
#so only the graph of Euclidean distance was shown. 
#By looking at the graph, we can say that close values are gathered together and therefore it is suitable for clustering.


### 4. Kümeleme

data.pca <- prcomp(pros, center = TRUE, scale. = TRUE)
data <- data.frame(PC1 = predict(data.pca)[,1])
 

# The raw data was divided into its principal components as center = TRUE and scale = TRUE 
# (with the Z-Score standardization method, the data was standardized to have means of 0 and standard deviations of 1.)
 
fviz_nbclust(data,kmeans,method = "wss", nstart=25) #for total within sum of square
fviz_nbclust(data,kmeans,method = "silhouette") #for average silhouette width
fviz_nbclust(data, kmeans, nstart = 25,  method = "gap_stat", nboot = 500) #or gap statistics
set.seed(123)
km_res <- kmeans(data, 3, nstart=25) 
 

# Since there was no serious outlier problem in the data, the k-means clustering method was chosen.

# The gap statistic graph said that there should be 1 cluster. By looking at the total within sum of square and silhouette graphics, 
it was concluded that the countries should be divided into 3 different clusters. Clustering was done with 3 clusters.
### 5. Interpretation of the Clusters

 
# Plot
cluster_colors <- c("green", "blue", "red")

plot(pca_pros_final, rep(0, length(pca_pros_final)), 
     xlab = "PC1", ylab = "", main = "K-means Clusters with PC1",
     col = cluster_colors[km_res$cluster], pch = 19)

legend("topright", legend = levels(as.factor(km_res$cluster)), 
       col = cluster_colors, pch = 19, title = "Cluster")
 

 
print(km_res)
 

# When we look at the details of the k-means clustering analysis with the km_res function, 
# we can see that a total of 47 countries are in the 1st cluster, 44 countries are in the 2nd cluster, 
# and 68 countries are in the 4th cluster. The 1st cluster includes countries such as Lebanon, Ghana and Tanzania, 
# the 2nd cluster includes countries such as Denmark and Germany, and the 3rd cluster includes countries such as Turkey, 
# Romania and Qatar. In this respect, we can say that we have grouped the countries in the world as "first, second and third world countries".
# 
# The total variance explained by clustering is 87.6%, which is quite high. 
# This shows that the cluster analysis is successful and provides a high amount of inter-cluster heterogeneity and intra-cluster homogeneity.

summary(pros)
aggregate(pros, by=list(cluster=km_res$cluster), mean)

# - If we compare all the variables (aggregate) in the 1st Cluster (African countries, etc.) with the summaries (pros) of the dataset, 
# it is even below the first quarter.
# 
# - When we compare all the variables in the 2nd cluster (Germany, Denmark, etc.) with the summaries of the dataset, 
# we see that they remain above the third quartile, on the contrary of the 1st cluster.
# 
# - When we look at the 3rd Cluster (Turkey, Romania, etc.), 
# we see that it has values close to the averages and medians we see in the summaries of the dataset, 
# and thus we can say that it has a structure that is in between the other two clusters.
