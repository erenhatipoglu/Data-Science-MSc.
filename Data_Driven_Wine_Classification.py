import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap


wine = pd.read_csv("https://archive.ics.uci.edu/static/public/109/data.csv")

## Descriptive Statistics

wine.head()
wine.info()
wine.shape
wine.describe()

# Histograms
numeric_cols = wine.select_dtypes(include=np.number).columns.drop(["class"])

n_cols = 2
n_rows = (len(numeric_cols) + 1) // 2

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 3 * n_rows))

axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(wine[col], kde=True, ax=axes[i], color='maroon')
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Boxplot
sns.boxplot(wine,orient="h")
plt.xticks(rotation=90,)

# "Proline" dominates other boxplot therefore we remove it temporarily
wine_no_pro = wine.drop(["Proline"], axis=1)
sns.boxplot(wine_no_pro,orient="h")
plt.xticks(rotation=90,)

# Corrplot
corrplot = wine.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corrplot, annot=True, cmap="BuGn", fmt=".2f",linewidths=.5)
plt.title("Correlation Matrix")
plt.show()

# Multicollinearity Check
cols = wine[['Alcohol', 'Malicacid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 
                       'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 
                       'Proanthocyanins', 'Color_intensity', 'Hue', 
                       '0D280_0D315_of_diluted_wines', 'Proline']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = cols.columns

# Calculating the VIF
vif_data["VIF"] = [variance_inflation_factor(cols.values, i)
                   for i in range(len(cols.columns))]

print(vif_data) # Multicollinearity looks insane but PCA will reduce it


## Principal Component Analysis

# Scaling the dataset before Principal Component Analysis
# ...and dropping the target variable
wine_classless = wine.drop(["class"], axis=1)

variables = wine_classless.columns

variables_standardized = StandardScaler().fit_transform(wine_classless)

# To check if its mean is 0 and standard deviation is 1
np.mean(variables_standardized), np.std(variables_standardized)

wine_standardized = pd.DataFrame(variables_standardized, columns = variables)

wine_standardized.head()

pca = PCA(n_components=len(wine_standardized.columns))
pca.fit(wine_standardized)

print("Eigen Values:", pca.explained_variance_)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# I will carry on with 3 components since they explain about 70% of the variance
pca = PCA(n_components=3)
pca.fit(wine_standardized)

# 2D Biplot
arrow_scaling_factor = 6
plt.figure(figsize=(14, 9))

for i, feature in enumerate(variables):
    plt.arrow(0, 0, pca.components_[0, i] * arrow_scaling_factor, pca.components_[1, i] * arrow_scaling_factor,
              head_width=0.1, head_length=0.1)
    plt.text(pca.components_[0, i] * arrow_scaling_factor * 1.15, pca.components_[1, i] * arrow_scaling_factor * 1.15,
             feature, fontsize=14)

PC_scores = pca.transform(wine_standardized)

sns.scatterplot(x=PC_scores[:, 0], y=PC_scores[:, 1], hue=wine["class"], palette="viridis")
plt.xlabel('PC1', fontsize=14)
plt.ylabel('PC2', fontsize=14)
plt.title('PCA Biplot - PC1 vs PC2', fontsize=16)

for i, label in enumerate(wine_standardized.index):
    plt.text(PC_scores[i, 0], PC_scores[i, 1], str(label), fontsize=8)

plt.show()

## Logistic Regression

## Classification with Logistic Regression before PCA
X = wine.drop(["class"], axis=1)
y = wine["class"]

# Encoding the target variable since it is categorical
y = (y > np.median(y)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Logistic Regression itself
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

print(conf_matrix)


## Classification after PCA
X = wine.drop(["class"], axis=1)
y = wine["class"]

y = (y > np.median(y)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

pca = PCA(n_components=3)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

print(conf_matrix) # All metrics are improved to 100%

# Scatter plot 
X_set, y_set = X_test, y_test
 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                     stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                     stop = X_set[:, 1].max() + 1, step = 0.01))
 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
 
plt.title('Logistic Regression (Test set)') 
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend()

plt.show()
