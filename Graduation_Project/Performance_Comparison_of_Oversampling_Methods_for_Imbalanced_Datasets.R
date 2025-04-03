library(caret)
library(tidyverse)
library(smotefamily)
library(ROSE)
library(nnet)
library(xgboost)
library(naivebayes)
library(ranger)
library(knitr)
library(corrplot)
library(ggplot2)
library(reshape2)


# Load data
heart <- read.csv("C:/Users/erenh/Desktop/deu donem projesi/heart.csv")

# Convert target to factor and set "1" as the positive class

heart$target <- as.factor(heart$target)
heart$target <- relevel(heart$target, ref = "1")
table(heart$target)


categorical_cols <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca")

# heart[categorical_cols] <- lapply(heart[categorical_cols], as.factor)

summary(heart)

dummies <- dummyVars(~ ., data = heart[categorical_cols])
one_hot_data <- predict(dummies, newdata = heart[categorical_cols])
heart <- cbind(heart[ , !(names(heart) %in% categorical_cols)], one_hot_data)


# Remove categorical variables from correlation matrix calculation
numeric_cols <- setdiff(names(heart), categorical_cols)
numeric_heart <- heart[numeric_cols]

summary(heart[numeric_cols])

# Compute correlation matrix
cor_matrix <- cor(numeric_heart[sapply(numeric_heart, is.numeric)], use = "complete.obs")

# Plot correlation matrix
corrplot(cor_matrix, method = "circle", type = "lower")

# Boxplots
# Identify numerical columns (excluding categorical columns and target)
numerical_cols <- setdiff(names(heart), c(categorical_cols, "target"))

# Scale the numerical columns
heart_scaled <- heart %>%
  select(all_of(numerical_cols)) %>%
  mutate(across(everything(), scale))

# Melt the scaled dataset for ggplot2
heart_melted <- heart_scaled %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# Create boxplots for scaled values
ggplot(heart_melted, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue", color = "black") +
  theme_minimal() +
  labs(title = "Olceklenmis Surekli Degiskenlerin Kutu Grafikleri",
       x = "Degiskenler",
       y = "Olceklenmiş Degerler") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability




set.seed(123) 
create_imbalanced <- function(majority, minority, ratio) {
  minority_sample <- minority %>% sample_frac(ratio)
  bind_rows(majority, minority_sample)
}

class_0 <- heart %>% filter(target == 0)
class_1 <- heart %>% filter(target == 1)

heart_5_imbalanced <- create_imbalanced(class_0, class_1, 0.05)
heart_15_imbalanced <- create_imbalanced(class_0, class_1, 0.15)
heart_25_imbalanced <- create_imbalanced(class_0, class_1, 0.25)

summary(heart_5_imbalanced$target)
summary(heart_15_imbalanced$target)
summary(heart_25_imbalanced$target)
summary(heart$target)

set.seed(123)
apply_resampling <- function(dataset) {
  X <- dataset[, -which(names(dataset) == "target")]
  y <- dataset$target
  
  # Ensure X is numeric and y is a factor
  X <- X[, sapply(X, is.numeric)]
  y <- as.factor(y)
  
  resampled <- list()
  
  # Adaptive ADASYN
  tryCatch({
    set.seed(123)
    target_ratio <- 0.5  # Desired ratio of minority class
    K <- 5               # Start with a default K for nearest neighbors
    max_iter <- 10       # Limit iterations
    for (i in 1:max_iter) {
      adasyn_result <- ADAS(X, y, K = K)
      heart_adasyn <- adasyn_result$data
      heart_adasyn$class <- as.factor(heart_adasyn$class)
      
      class_counts <- table(heart_adasyn$class)
      balance_ratio <- class_counts[2] / sum(class_counts)
      
      if (abs(balance_ratio - target_ratio) < 0.05) {
        resampled$ADASYN <- heart_adasyn
        break
      }
      
      K <- K + 1  # Adjust K to use more neighbors in subsequent iterations
    }
  }, error = function(e) { message("ADASYN failed: ", e$message) })
  
  
  # Borderline-SMOTE with Adaptive Parameters
  tryCatch({
    set.seed(123)
    target_ratio <- 0.5
    C <- 15
    max_iter <- 10
    for (i in 1:max_iter) {
      borderline_smote_result <- BLSMOTE(X, y, K = 5, C = C)
      heart_borderline_smote <- borderline_smote_result$data
      heart_borderline_smote$class <- as.factor(heart_borderline_smote$class)
      
      class_counts <- table(heart_borderline_smote$class)
      balance_ratio <- class_counts[2] / sum(class_counts)
      
      if (abs(balance_ratio - target_ratio) < 0.05) {
        resampled$Borderline_SMOTE <- heart_borderline_smote
        break
      }
      
      C <- C + 1  # Adjust C parameter
    }
  }, error = function(e) { message("Borderline-SMOTE failed: ", e$message) })
  
  # RUS with Adaptive Parameters
  tryCatch({
    set.seed(123)
    target_ratio <- 0.5
    min_size <- min(table(y))
    for (i in seq(min_size, length(y) / 2, by = 1)) {
      heart_rus <- dataset %>%
        group_by(target) %>%
        slice_sample(n = i) %>%
        ungroup()
      
      class_counts <- table(heart_rus$target)
      balance_ratio <- class_counts[2] / sum(class_counts)
      
      if (abs(balance_ratio - target_ratio) < 0.05) {
        resampled$RUS <- heart_rus
        break
      }
    }
  }, error = function(e) { message("RUS failed: ", e$message) })
  
  # ROSE with Adaptive Parameters
  tryCatch({
    set.seed(123)
    target_ratio <- 0.5
    max_iter <- 10
    rose_iterations <- seq(1, max_iter, by = 1)
    for (iter in rose_iterations) {
      heart_rose <- ROSE(target ~ ., data = dataset, seed = iter)$data
      heart_rose$target <- as.factor(heart_rose$target)
      
      class_counts <- table(heart_rose$target)
      balance_ratio <- class_counts[2] / sum(class_counts)
      
      if (abs(balance_ratio - target_ratio) < 0.05) {
        resampled$ROSE <- heart_rose
        break
      }
    }
  }, error = function(e) { message("ROSE failed: ", e$message) })
  
  # SMOTE with Adaptive Parameters (updated for fallback)
  tryCatch({
    set.seed(123)
    target_ratio <- 0.5
    dup_size <- 1
    max_iter <- 10
    last_smote <- NULL
    for (i in 1:max_iter) {
      smote_result <- SMOTE(X, y, K = 1, dup_size = dup_size)
      heart_smote <- smote_result$data
      heart_smote$class <- as.factor(heart_smote$class)
      last_smote <- heart_smote  # Save the latest result
      
      class_counts <- table(heart_smote$class)
      balance_ratio <- class_counts[2] / sum(class_counts)
      
      if (abs(balance_ratio - target_ratio) < 0.05) {
        resampled$SMOTE <- heart_smote
        break
      }
      
      dup_size <- dup_size + 1
    }
    # Add the last available SMOTE result if balance wasn't achieved
    if (is.null(resampled$SMOTE)) {
      resampled$SMOTE <- last_smote
    }
  }, error = function(e) { message("SMOTE failed: ", e$message) })
  
  return(resampled)
}

# Apply Resampling to Imbalanced Datasets
datasets <- list(
  "Original" = heart,
  "5% Imbalance" = heart_5_imbalanced,
  "15% Imbalance" = heart_15_imbalanced,
  "25% Imbalance" = heart_25_imbalanced
)

resampling_results <- lapply(datasets[-1], apply_resampling)  # Apply resampling only to imbalanced datasets
resampling_results$Original <- list(Original = heart)  # Add the original dataset

print(names(resampling_results$`5% Imbalance`))


# Print counts of 0's and 1's in each resampled dataset
for (dataset_name in names(resampling_results)) {
  cat("\nDataset:", dataset_name, "\n")
  
  for (method_name in names(resampling_results[[dataset_name]])) {
    cat("  Method:", method_name, "\n")
    
    # Determine the column name for the target variable
    target_col <- ifelse(method_name %in% c("RUS", "ROSE"), "target", "class")
    
    # Print counts
    table_counts <- table(resampling_results[[dataset_name]][[method_name]][[target_col]])
    print(table_counts)
  }
}



# Helper function to extract features and target
data_split <- function(data, target_name) {
  target <- as.factor(data[[target_name]])
  features <- data[, !(names(data) %in% target_name)]
  return(list(X = features, y = target))
}

# Function to apply models to imbalanced datasets
apply_models <- function(dataset, target_name) {
  set.seed(123)
  # Split data into features and target
  split_data <- data_split(dataset, target_name)
  X <- split_data$X
  y <- split_data$y
  
  # Train-test split
  set.seed(123)
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  y_train <- y[train_index]
  X_test <- X[-train_index, ]
  y_test <- y[-train_index]
  
  results <- list()
  
  # Logistic Regression
  tryCatch({
    set.seed(123)
    lr_model <- train(X_train, y_train, method = "glm", family = "binomial",
                      trControl = trainControl(method = "cv", number = 3))
    results$Logistic_Regression <- list(model = lr_model, 
                                        test_predictions = predict(lr_model, X_test), 
                                        test_actual = y_test)
  }, error = function(e) { message("Logistic Regression failed: ", e$message) })
  
  # ANN
  tryCatch({
    set.seed(123)
    ann_model <- train(X_train, y_train, method = "nnet", trace = FALSE,
                       trControl = trainControl(method = "cv", number = 3))
    results$ANN <- list(model = ann_model, 
                        test_predictions = predict(ann_model, X_test), 
                        test_actual = y_test)
  }, error = function(e) { message("ANN failed: ", e$message) })
  
  # Naive Bayes
  tryCatch({
    set.seed(123)
    nb_model <- train(X_train, y_train, method = "naive_bayes",
                      trControl = trainControl(method = "cv", number = 3))
    results$Naive_Bayes <- list(model = nb_model, 
                                test_predictions = predict(nb_model, X_test), 
                                test_actual = y_test)
  }, error = function(e) { message("Naive Bayes failed: ", e$message) })
  
  # Random Forest
  tryCatch({
    set.seed(123)
    rf_model <- train(X_train, y_train, method = "ranger",
                      trControl = trainControl(method = "cv", number = 3))
    results$Random_Forest <- list(model = rf_model, 
                                  test_predictions = predict(rf_model, X_test), 
                                  test_actual = y_test)
  }, error = function(e) { message("Random Forest failed: ", e$message) })
  
  # SVM
  tryCatch({
    set.seed(123)
    svm_model <- train(X_train, y_train, method = "svmRadial",
                       trControl = trainControl(method = "cv", number = 3))
    results$SVM <- list(model = svm_model, 
                        test_predictions = predict(svm_model, X_test), 
                        test_actual = y_test)
  }, error = function(e) { message("SVM failed: ", e$message) })
  
  # XGBoost
  tryCatch({
    set.seed(123)
    xgb_model <- train(X_train, y_train, method = "xgbTree",
                       trControl = trainControl(method = "cv", number = 3))
    results$XGBoost <- list(model = xgb_model, 
                            test_predictions = predict(xgb_model, X_test), 
                            test_actual = y_test)
  }, error = function(e) { message("XGBoost failed: ", e$message) })
  
  return(results)
}


# Iterate over datasets and resampling methods
final_results <- list()

for (dataset_name in names(resampling_results)) {
  final_results[[dataset_name]] <- list()
  for (method_name in names(resampling_results[[dataset_name]])) {
    dataset <- resampling_results[[dataset_name]][[method_name]]
    target_name <- ifelse(method_name %in% c("RUS", "ROSE", "Original"), "target", "class")
    message("Processing: ", dataset_name, " - ", method_name)
    final_results[[dataset_name]][[method_name]] <- apply_models(dataset, target_name)
  }
}

# Display results for each dataset and method
final_results


# Function to calculate performance metrics
calculate_metrics <- function(predictions, actual) {
  # Confusion Matrix
  conf_mat <- confusionMatrix(predictions, actual)
  
  # Extract Metrics
  accuracy <- conf_mat$overall['Accuracy']
  balanced_acc <- conf_mat$byClass['Balanced Accuracy']
  sensitivity <- conf_mat$byClass['Sensitivity']
  specificity <- conf_mat$byClass['Specificity']
  
  # F1 Score
  precision <- conf_mat$byClass['Precision']
  recall <- conf_mat$byClass['Recall']
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  return(data.frame(
    Accuracy = accuracy,
    Balanced_Accuracy = balanced_acc,
    Sensitivity = sensitivity,
    Specificity = specificity,
    F1_Score = f1_score
  ))
}

# Initialize a list to store metrics for resampled datasets
resampled_metrics <- list()

# Compute metrics for each resampled dataset and method
for (dataset_name in names(resampling_results)) {
  resampled_metrics[[dataset_name]] <- list()
  for (method_name in names(resampling_results[[dataset_name]])) {
    models <- final_results[[dataset_name]][[method_name]]
    for (model_name in names(models)) {
      model_results <- models[[model_name]]
      if (!is.null(model_results)) {
        # Extract predictions and actuals
        test_predictions <- model_results$test_predictions
        test_actual <- model_results$test_actual
        
        # Calculate metrics
        metrics <- calculate_metrics(test_predictions, test_actual)
        
        # Store results in a structured format
        metrics$Dataset <- dataset_name
        metrics$Resampling_Method <- method_name
        metrics$Model <- model_name
        resampled_metrics[[dataset_name]][[method_name]][[model_name]] <- metrics
      }
    }
  }
}

# Convert the list to a data frame for visualization
df_resampled_metrics <- do.call(rbind, lapply(names(resampled_metrics), function(dataset_name) {
  do.call(rbind, lapply(names(resampled_metrics[[dataset_name]]), function(method_name) {
    do.call(rbind, lapply(names(resampled_metrics[[dataset_name]][[method_name]]), function(model_name) {
      metrics <- resampled_metrics[[dataset_name]][[method_name]][[model_name]]
      return(metrics)
    }))
  }))
}))

# Display the accuracy results in a table
# Initialize a data frame to store accuracy results
accuracy_results <- data.frame(
  Dataset = character(),
  Resampling_Method = character(),
  Model = character(),
  Metric = numeric(),
  stringsAsFactors = FALSE
)

# Populate the accuracy_results data frame
for (dataset_name in names(final_results)) {
  for (method_name in names(final_results[[dataset_name]])) {
    for (model_name in names(final_results[[dataset_name]][[method_name]])) {
      model_results <- final_results[[dataset_name]][[method_name]][[model_name]]
      if (!is.null(model_results)) {
        # Extract predictions and actuals
        test_predictions <- model_results$test_predictions
        test_actual <- model_results$test_actual
        
        # Calculate accuracy
        accuracy <- mean(test_predictions == test_actual)
        
        # Append to the accuracy_results data frame
        accuracy_results <- rbind(accuracy_results, data.frame(
          Dataset = dataset_name,
          Resampling_Method = method_name,
          Model = model_name,
          Metric = accuracy,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

# Display the accuracy results
print(accuracy_results)

# Proceed to create plots based on accuracy_results


# Rename columns for easier mapping in ggplot
colnames(accuracy_results) <- c("Dataset", "Resampling_Method", "Algorithm", "Metric")

# Split the data for Original and Imbalanced datasets
original_data <- accuracy_results %>% filter(Dataset == "Original")
imbalanced_data <- accuracy_results %>% filter(Dataset != "Original")

# First create df_metrics for the original dataset
metrics_results_original <- list()

# Calculate metrics for the original dataset
for (model_name in names(final_results$Original$Original)) {
  model_results <- final_results$Original$Original[[model_name]]
  if (!is.null(model_results)) {
    # Extract predictions and actuals
    test_predictions <- model_results$test_predictions
    test_actual <- model_results$test_actual
    
    # Calculate confusion matrix
    conf_mat <- confusionMatrix(test_predictions, test_actual)
    
    # Extract metrics
    accuracy <- conf_mat$overall['Accuracy']
    balanced_acc <- conf_mat$byClass['Balanced Accuracy']
    sensitivity <- conf_mat$byClass['Sensitivity']
    specificity <- conf_mat$byClass['Specificity']
    
    # Calculate F1 Score
    precision <- conf_mat$byClass['Precision']
    recall <- conf_mat$byClass['Recall']
    f1_score <- 2 * (precision * recall) / (precision + recall)
    
    # Store results
    metrics_results_original[[model_name]] <- data.frame(
      Model = model_name,
      Accuracy = accuracy,
      Balanced_Accuracy = balanced_acc,
      Sensitivity = sensitivity,
      Specificity = specificity,
      F1_Score = f1_score
    )
  }
}

# Combine all results into one data frame
df_metrics_original <- do.call(rbind, metrics_results_original)

# Reshape the data for plotting
original_metrics_long <- df_metrics_original %>%
  pivot_longer(
    cols = c("Accuracy", "Sensitivity", "Specificity","Balanced_Accuracy", "F1_Score"),
    names_to = "Metric_Type",
    values_to = "Value"
  )


# Original Dataset Plot
ggplot(original_metrics_long, aes(x = Model, y = Value, fill = Metric_Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  theme_minimal() +
  labs(
    title = "Model Performance Metrics on Original Dataset",
    x = "Algorithm",
    y = "Score",
    fill = "Metric"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1))


imbalanced_data$Dataset <- factor(
  imbalanced_data$Dataset, 
  levels = c("5% Imbalance", "15% Imbalance", "25% Imbalance")
)


# Imbalanced Dataset Plot
ggplot(imbalanced_data, aes(x = Algorithm, y = as.numeric(Metric), fill = Resampling_Method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  facet_wrap(~ Dataset, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Model Accuracy Across Imbalanced Datasets and Resampling Methods",
    x = "Algorithm",
    y = "Accuracy",
    fill = "Resampling Method"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  ) +
  scale_fill_brewer(palette = "Set1")

# Convert df_resampled_metrics to long format for visualization
df_resampled_metrics_long <- df_resampled_metrics %>%
  pivot_longer(
    cols = c("Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1_Score"),
    names_to = "Metric_Type",
    values_to = "Value"
  )


# Ensure the Dataset column is ordered as 5%, 15%, and 25% Imbalance
df_resampled_metrics_long$Dataset <- factor(
  df_resampled_metrics_long$Dataset, 
  levels = c("5% Imbalance", "15% Imbalance", "25% Imbalance")
)

# Filter out the original dataset
df_resampled_metrics_long_filtered <- df_resampled_metrics_long %>%
  filter(Dataset %in% c("5% Imbalance", "15% Imbalance", "25% Imbalance"))

# Balanced Accuracy Plot
ggplot(df_resampled_metrics_long_filtered %>% filter(Metric_Type == "Balanced_Accuracy"), 
       aes(x = Model, y = Value, fill = Resampling_Method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  facet_wrap(~ Dataset, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Model Balanced Accuracy Across Resampled Datasets",
    x = "Algorithm",
    y = "Balanced Accuracy",
    fill = "Resampling Method"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1))

# Sensitivity Plot
ggplot(df_resampled_metrics_long_filtered %>% filter(Metric_Type == "Sensitivity"), 
       aes(x = Model, y = Value, fill = Resampling_Method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  facet_wrap(~ Dataset, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Model Sensitivity Across Resampled Datasets",
    x = "Algorithm",
    y = "Sensitivity",
    fill = "Resampling Method"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1))

# Specificity Plot
ggplot(df_resampled_metrics_long_filtered %>% filter(Metric_Type == "Specificity"), 
       aes(x = Model, y = Value, fill = Resampling_Method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  facet_wrap(~ Dataset, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Model Specificity Across Resampled Datasets",
    x = "Algorithm",
    y = "Specificity",
    fill = "Resampling Method"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1))

# F1 Score Plot
ggplot(df_resampled_metrics_long_filtered %>% filter(Metric_Type == "F1_Score"), 
       aes(x = Model, y = Value, fill = Resampling_Method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  facet_wrap(~ Dataset, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Model F1 Score Across Resampled Datasets",
    x = "Algorithm",
    y = "F1 Score",
    fill = "Resampling Method"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1))










## imbalanced datasetleri iC'in


# Initialize a list to store metrics for each combination
metrics_results <- list()

# Iterate over the final_results
for (dataset_name in names(final_results)) {
  metrics_results[[dataset_name]] <- list()
  for (method_name in names(final_results[[dataset_name]])) {
    models <- final_results[[dataset_name]][[method_name]]
    for (model_name in names(models)) {
      model_results <- models[[model_name]]
      if (!is.null(model_results)) {
        # Extract predictions and actuals
        test_predictions <- model_results$test_predictions
        test_actual <- model_results$test_actual
        
        # Calculate metrics
        metrics <- calculate_metrics(test_predictions, test_actual)
        
        # Append results to the list
        metrics_results[[dataset_name]][[method_name]][[model_name]] <- metrics
      }
    }
  }
}

# Convert the list to a data frame for kable
df_metrics <- do.call(rbind, lapply(names(metrics_results), function(dataset_name) {
  do.call(rbind, lapply(names(metrics_results[[dataset_name]]), function(method_name) {
    do.call(rbind, lapply(names(metrics_results[[dataset_name]][[method_name]]), function(model_name) {
      result <- metrics_results[[dataset_name]][[method_name]][[model_name]]
      result$Dataset <- dataset_name
      result$Resampling_Method <- method_name
      result$Model <- model_name
      return(result)
    }))
  }))
}))

# Display results in a kable table
kable(df_metrics, format = "markdown", 
      col.names = c("Dataset", "Resampling Method", "Model", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "F1 Score", "AUC"),
      digits = 3, 
      caption = "Performance Metrics of Models Across Datasets and Resampling Methods")


# Additional Analysis on Imbalanced Datasets

# Use the previously created imbalanced datasets
imbalanced_datasets <- list(
  "5% Imbalance" = heart_5_imbalanced,
  "15% Imbalance" = heart_15_imbalanced,
  "25% Imbalance" = heart_25_imbalanced
)

# Function to apply models to imbalanced datasets
apply_models_to_imbalanced <- function(dataset, target_name) {
  set.seed(123)
  # Split data into features and target
  split_data <- data_split(dataset, target_name)
  X <- split_data$X
  y <- split_data$y
  
  # Train-test split
  set.seed(123)
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  y_train <- y[train_index]
  X_test <- X[-train_index, ]
  y_test <- y[-train_index]
  
  results <- list()
  
  # Logistic Regression
  tryCatch({
    set.seed(123)
    lr_model <- train(X_train, y_train, method = "glm", family = "binomial",
                      trControl = trainControl(method = "cv", number = 3))
    results$Logistic_Regression <- list(model = lr_model, 
                                        test_predictions = predict(lr_model, X_test), 
                                        test_actual = y_test)
  }, error = function(e) { message("Logistic Regression failed: ", e$message) })
  
  # ANN
  tryCatch({
    set.seed(123)
    ann_model <- train(X_train, y_train, method = "nnet", trace = FALSE,
                       trControl = trainControl(method = "cv", number = 3))
    results$ANN <- list(model = ann_model, 
                        test_predictions = predict(ann_model, X_test), 
                        test_actual = y_test)
  }, error = function(e) { message("ANN failed: ", e$message) })
  
  # Naive Bayes
  tryCatch({
    set.seed(123)
    nb_model <- train(X_train, y_train, method = "naive_bayes",
                      trControl = trainControl(method = "cv", number = 3))
    results$Naive_Bayes <- list(model = nb_model, 
                                test_predictions = predict(nb_model, X_test), 
                                test_actual = y_test)
  }, error = function(e) { message("Naive Bayes failed: ", e$message) })
  
  # Random Forest
  tryCatch({
    set.seed(123)
    rf_model <- train(X_train, y_train, method = "ranger",
                      trControl = trainControl(method = "cv", number = 3))
    results$Random_Forest <- list(model = rf_model, 
                                  test_predictions = predict(rf_model, X_test), 
                                  test_actual = y_test)
  }, error = function(e) { message("Random Forest failed: ", e$message) })
  
  # SVM
  tryCatch({
    set.seed(123)
    svm_model <- train(X_train, y_train, method = "svmRadial",
                       trControl = trainControl(method = "cv", number = 3))
    results$SVM <- list(model = svm_model, 
                        test_predictions = predict(svm_model, X_test), 
                        test_actual = y_test)
  }, error = function(e) { message("SVM failed: ", e$message) })
  
  # XGBoost
  tryCatch({
    set.seed(123)
    xgb_model <- train(X_train, y_train, method = "xgbTree",
                       trControl = trainControl(method = "cv", number = 3))
    results$XGBoost <- list(model = xgb_model, 
                            test_predictions = predict(xgb_model, X_test), 
                            test_actual = y_test)
  }, error = function(e) { message("XGBoost failed: ", e$message) })
  
  return(results)
}

# Process imbalanced datasets
imbalanced_results <- list()
for (dataset_name in names(imbalanced_datasets)) {
  dataset <- imbalanced_datasets[[dataset_name]]
  target_name <- "target"
  message("Processing: ", dataset_name)
  imbalanced_results[[dataset_name]] <- apply_models_to_imbalanced(dataset, target_name)
}

# Compute metrics for imbalanced datasets
imbalanced_metrics_results <- list()

for (dataset_name in names(imbalanced_results)) {
  imbalanced_metrics_results[[dataset_name]] <- list()
  for (model_name in names(imbalanced_results[[dataset_name]])) {
    model_results <- imbalanced_results[[dataset_name]][[model_name]]
    if (!is.null(model_results)) {
      # Extract predictions and actuals
      test_predictions <- model_results$test_predictions
      test_actual <- model_results$test_actual
      
      # Calculate metrics
      metrics <- calculate_metrics(test_predictions, test_actual)
      
      # Append results 
      imbalanced_metrics_results[[dataset_name]][[model_name]] <- metrics
    }
  }
}

# Convert the list to a data frame for kable
df_imbalanced_metrics <- do.call(rbind, lapply(names(imbalanced_metrics_results), function(dataset_name) {
  do.call(rbind, lapply(names(imbalanced_metrics_results[[dataset_name]]), function(model_name) {
    result <- imbalanced_metrics_results[[dataset_name]][[model_name]]
    result$Dataset <- dataset_name
    result$Model <- model_name
    return(result)
  }))
}))

# Display results in a kable table
kable(df_imbalanced_metrics, format = "markdown", 
      col.names = c("Dataset", "Model", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "F1 Score", "AUC"),
      digits = 3, 
      caption = "Performance Metrics of Models on Imbalanced Datasets")

# Plotting for Imbalanced Datasets
imbalanced_accuracy_results <- data.frame(
  Dataset = character(),
  Model = character(),
  Metric = numeric(),
  stringsAsFactors = FALSE
)

for (dataset_name in names(imbalanced_results)) {
  for (model_name in names(imbalanced_results[[dataset_name]])) {
    model_results <- imbalanced_results[[dataset_name]][[model_name]]
    if (!is.null(model_results)) {
      # Extract predictions and actuals
      test_predictions <- model_results$test_predictions
      test_actual <- model_results$test_actual
      
      # Calculate accuracy
      accuracy <- mean(test_predictions == test_actual)
      
      # Append results to the data frame
      imbalanced_accuracy_results <- rbind(imbalanced_accuracy_results, data.frame(
        Dataset = dataset_name,
        Model = model_name,
        Metric = accuracy,
        stringsAsFactors = FALSE
      ))
    }
  }
}


# Split the data for Original and Imbalanced datasets
original_accuracy_results <- imbalanced_accuracy_results %>% filter(Dataset == "Original")
imbalanced_accuracy_results_filtered <- imbalanced_accuracy_results %>% filter(Dataset != "Original")



# Create long-format data for plotting
imbalanced_metrics_long <- df_imbalanced_metrics %>%
  pivot_longer(
    cols = c("Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1_Score"),
    names_to = "Metric_Type",
    values_to = "Value"
  )


# Combine the datasets for 5%, 15%, and 25% imbalance into one data frame
combined_imbalanced_metrics_long <- imbalanced_metrics_long %>%
  filter(Dataset %in% c("5% Imbalance", "15% Imbalance", "25% Imbalance"))

# Ensure the Dataset column is ordered as 5%, 15%, and 25% Imbalance
combined_imbalanced_metrics_long$Dataset <- factor(
  combined_imbalanced_metrics_long$Dataset, 
  levels = c("5% Imbalance", "15% Imbalance", "25% Imbalance")
)

# Plot all datasets together using facet_wrap
ggplot(combined_imbalanced_metrics_long, 
       aes(x = Model, y = Value, fill = Metric_Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  facet_wrap(~ Dataset, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Model Metrics Across Imbalanced Datasets",
    x = "Model",
    y = "Score",
    fill = "Metric"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1))
