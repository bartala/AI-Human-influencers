# Install necessary packages
install.packages("caret")

# Load the library
library(caret)
setwd("C:\\your_path")
#text.df<-read.csv('data1.csv', stringsAsFactors=FALSE)
data <- read.csv("sentiment_analysis_results1.csv",  stringsAsFactors=FALSE)




# Convert label to factor
data$user_type <- as.factor(data$user_type)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$user_type, p = .8, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- data[ trainIndex,]
dataTest  <- data[-trainIndex,]

# Train a Logistic Regression model
model <- glm(user_type ~ posts.likes_count + pot.comment.likes_count + sentiment + popularity, data = dataTrain, family = binomial)

# Make predictions
predictions_prob <- predict(model, dataTest, type = "response")
predictions <- ifelse(predictions_prob > 0.5, "HUMAN", "AI")

# Convert predictions to factor
predictions <- as.factor(predictions)

# Evaluate the model
confusionMatrix(predictions, dataTest$user_type)

# Print model summary
summary(model)


# Load necessary libraries
library(caret)
library(pROC)
library(yardstick)

# Train a Logistic Regression model
model <- glm(user_type ~ posts.likes_count + pot.comment.likes_count + sentiment + popularity, data = dataTrain, family = binomial)

# Make predictions
predictions_prob <- predict(model, dataTest, type = "response")
predictions <- ifelse(predictions_prob > 0.5, "HUMAN", "AI")

# Convert predictions to factor
predictions <- as.factor(predictions)

# Convert the true labels to factor for confusionMatrix
dataTest$user_type <- as.factor(dataTest$user_type)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, dataTest$user_type)
print(conf_matrix)
# Calculate F1 score using the yardstick package
f1_score <- f_meas_vec(dataTest$user_type, predictions, estimator = "binary", event_level = "second") # Assuming 'AI' is the positive class
print(paste("F1 Score:", round(f1_score, 3)))

# Calculate F1 score manually
precision <- conf_matrix$byClass['Pos Pred Value'] # Same as Precision
recall <- conf_matrix$byClass['Sensitivity']       # Same as Recall
f1_manual <- 2 * ((precision * recall) / (precision + recall))
print(paste("Manual F1 Score:", round(f1_manual, 3)))

# Plot ROC curve using pROC package
roc_obj <- roc(dataTest$user_type, predictions_prob, levels = rev(levels(dataTest$user_type)))
plot(roc_obj, col = "blue", main = "ROC Curve for Logistic Regression")
auc_value <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)

# Load necessary library
library(pROC)

# Calculate ROC object
roc_obj <- roc(dataTest$user_type, predictions_prob, levels = rev(levels(dataTest$user_type)))

# Plot ROC curve with custom labels for TPR and FPR
plot(roc_obj, col = "blue", main = "ROC Curve for Logistic Regression", 
     xlab = "False Positive Rate ", 
     ylab = "True Positive Rate ")

# Calculate AUC
auc_value <- auc(roc_obj)

# Add AUC to the plot
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)


# Print model summary
summary(model)


roc_curve<- roc(dataTest$user_type,predictions_prob)
sensitivity <- roc_curve$sensitivities
specificity<- roc_curve$specificities
reversed_spec=1-specificity
plot(reversed_spec,sensitivity,type="l",col="blue",lwd=2,xlab="FPR",ylab="TPR",main="roc curve")
abline(a=0,b=1,lty=2,col="red")




###### model with words



# Install necessary packages
install.packages("text")
install.packages("tidyverse")
install.packages("caret")

# Load the libraries
library(text)
library(tidyverse)
library(caret)

# Load your dataset with proper encoding

setwd("C:\\your_path")
#text.df<-read.csv('seminar1.csv', stringsAsFactors=FALSE)
data <- read.csv("sentiment_analysis_results1.csv",  stringsAsFactors=FALSE)

# Convert labels to factors
data$user_type <- as.factor(data$user_type)

# Extract BERT embeddings for the comments
bert_embeddings <- textEmbed(texts = data$text.comments.text, model = 'bert-base-uncased')

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$label, p = 0.8, list = FALSE)
trainData <- bert_embeddings[trainIndex,]
testData <- bert_embeddings[-trainIndex,]

trainLabels <- data$label[trainIndex]
testLabels <- data$label[-trainIndex]

# Train a logistic regression model using the BERT embeddings
model <- train(
  trainLabels ~ ., 
  data = as.data.frame(trainData), 
  method = "glm", 
  family = binomial()
)

# Predict on the test set
predictions <- predict(model, newdata = as.data.frame(testData))

# Evaluate the model
confusionMatrix(predictions, testLabels)


# Load necessary libraries
library(tm)  # for text mining
library(caret)
library(glmnet)  # for logistic regression
library(tidyr)  # for data manipulation

# Example data: replace this with your actual data

setwd("C:\\seminar")
#text.df<-read.csv('data1.csv', stringsAsFactors=FALSE)
data <- read.csv("sentiment_analysis_results1.csv",  stringsAsFactors=FALSE)

# Example data: replace this with your actual data
data <- data.frame(data)

# Convert label to factor
data$user_type <- as.factor(data$user_type)

# Text preprocessing and vectorization using TF-IDF
corpus <- Corpus(VectorSource(data$posts.comments.text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
dtm <- DocumentTermMatrix(corpus)
tfidf <- weightTfIdf(dtm)

# Convert to a data frame
comments_tfidf <- as.data.frame(as.matrix(tfidf))
comments_tfidf$label <- data$label

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(comments_tfidf$label, p = .8, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- comments_tfidf[trainIndex,]
dataTest  <- comments_tfidf[-trainIndex,]

# Train a Logistic Regression model
model <- glm(label ~ ., data = dataTrain[, -ncol(dataTrain)], family = binomial)

# Make predictions on test data
predictions <- predict(model, newdata = dataTest[, -ncol(dataTest)], type = "response")
predicted_labels <- ifelse(predictions > 0.5, "HUMAN", "AI")

# Evaluate the model (optional)
confusionMatrix(table(predicted_labels, dataTest$label))

# Example of predicting a new text comment
new_comment <- "AI is advancing rapidly"
new_comment_tfidf <- as.data.frame(as.matrix(weightTfIdf(DocumentTermMatrix(Corpus(VectorSource(new_comment)), control = list(dictionary = Terms(tfidf))))))
prediction_new <- predict(model, newdata = new_comment_tfidf, type = "response")
predicted_class_new <- ifelse(prediction_new > 0.5, "HUMAN", "AI")

print(paste("Predicted class for new comment:", predicted_class_new))
