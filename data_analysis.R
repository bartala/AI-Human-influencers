library(dplyr)
library(stringr)
library(readxl)

setwd("C:\\your_dir")
text.df <- read_excel("data2.xlsx", sheet = "ALL")
head(text.df)

# cleaning the data

remove_usernames <- function(text) {
  str_remove_all(text, "@\\w+")
}

text.df <- text.df %>%mutate(cleaned_text = sapply(posts.comments.text, remove_usernames))



library(sentimentr)

# Function to analyze sentiment
analyze_sentiment <- function(text) {
  sentiment_score <- sentiment_by(text)$ave_sentiment
  
  if (sentiment_score > 0) {
    return(1)
  } else if (sentiment_score < 0) {
    return(-1)
  } else {
    return(0)
  }
}

# Apply the sentiment analysis function to the cleaned_text column and create a new sentiment column
text.df <- text.df %>%
  mutate(sentiment = sapply(cleaned_text, analyze_sentiment))



#write_xlsx(data, "output_file.xlsx")

text.df$text_length <- nchar(text.df$cleaned_text)
head(text.df)


write.csv(text.df, "sentiment_analysis_results1.csv", row.names = FALSE)


## T-TEST POPULARITY 
library(dplyr)

# Load the library
library(caret)
setwd("C:\\your_dir")
#text.df<-read.csv('seminar1.csv', stringsAsFactors=FALSE)
data <- read.csv("sentiment_analysis_results1.csv",  stringsAsFactors=FALSE)


new_df <- data %>% select ("user_type", "popularity")%>% distinct()

new_df$`user_type` <- as.factor(new_df$`user_type`)
# Perform t-test
t_test_result0 <- t.test(popularity ~ `user_type`, data = new_df)

# Print t-test result
print(t_test_result0)

ggplot(new_df, aes(x = user_type, y = popularity)) +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "point", shape = 20, size = 3, color = "red") +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "blue") +
  scale_x_discrete(labels = c("HUMAN" = "HUMAN", "AI" = "HUMANOID-AI")) +  # Manually set the labels
  labs(
    x = "User Type",
    y = "Popularity") +
  theme_minimal()

# Create a box plot to visualize the t-test results
ggplot(new_df, aes(x = user_type, y = popularity)) +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "point", shape = 20, size = 3, color = "red") +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "blue") +
  labs(
       x = "User Type",
       y = "length") +
  theme_minimal()

## T- TEST POST.LIKE.COUNT

# Assuming the DataFrame has columns 'user_type' and 'POST_LIKES.COUNT'
new_df_likes <- data %>% select(`user_type`, `posts.likes_count`) %>% distinct()
# Ensure the 'user_type' column is a factor
new_df_likes$`user_type` <- as.factor(new_df_likes$`user_type`)

# Perform t-test
t_test_result <- t.test(`posts.likes_count` ~ `user_type`, data = new_df_likes)

# Print t-test result
print(t_test_result)


ggplot(new_df_likes, aes(x = user_type, y = posts.likes_count)) +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "point", shape = 20, size = 3, color = "red") +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "blue") +
  scale_x_discrete(labels = c("HUMAN" = "HUMAN", "AI" = "HUMANOID-AI")) +  # Manually set the labels
  labs(
    x = "User Type",
    y = "Likes per Post") +
  theme_minimal()

## T-TEST likes per comment

new_df_likecomments <- data %>% select(`user_type`, `pot.comment.likes_count`)

# Ensure the 'user_type' column is a factor
new_df_likecomments$`user_type` <- as.factor(new_df_likecomments$`user_type`)

# Perform t-test
t_test_result1 <- t.test(`pot.comment.likes_count` ~ `user_type`, data = new_df_likecomments)

# Print t-test result
print(t_test_result1)


ggplot(new_df_likecomments, aes(x = user_type, y = pot.comment.likes_count)) +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "point", shape = 20, size = 3, color = "red") +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "blue") +
  scale_x_discrete(labels = c("HUMAN" = "HUMAN", "AI" = "HUMANOID-AI")) +  # Manually set the labels
  labs(
    x = "User Type",
    y = "Likes per Comment") +
  theme_minimal()


# using Wilcoxon Rank-Sum Test (Mann-Whitney U Test)
new_df_SENTIMENT <- text.df %>% select(`user_type`, `sentiment`)
new_df_SENTIMENT$`user_type` <- as.factor(new_df_SENTIMENT$`user_type`)

result <- wilcox.test(sentiment ~ `user_type`, data = new_df_SENTIMENT)

count_df <- text.df %>%
  group_by('user_type', sentiment) %>%
  summarise(count = n(), .groups = 'drop')

# Count occurrences of sentiment by user_type using table
sentiment_counts <- table(new_df_SENTIMENT$`user_type`, new_df_SENTIMENT$sentiment)

# Print the counts
print(sentiment_counts)

print(count_df)
print(result)

###### t test senitmnet 1

# Filter the DataFrame
filtered_df <- text.df[text.df$'user_type' %in% c('HUMAN', 'AI') & text.df$sentiment == 1,  c("user_type", "sentiment")]

# Print the filtered DataFrame
head(filtered_df)

new_df_SENTIMENT <- filtered_df %>% select(`user_type`, `sentiment`)
new_df_SENTIMENT$`user_type` <- as.factor(new_df_SENTIMENT$`user_type`)
t_test_sentiment <- t.test(`sentiment` ~ `user_type`, data = new_df_SENTIMENT)




## T TEST FOR LENGTH

setwd("C:\\your_dir")
text.df<-read.csv('new_dataset.csv', stringsAsFactors=FALSE)
#text.df <- read_excel("data2.xlsx", sheet = "ALL")

new_df_length <- text.df %>% select(`user_type`, comment_length)

# Ensure the 'user_type' column is a factor
new_df_length$`user_type` <- as.factor(new_df_length$`user_type`)

# Perform t-test
t_test_result3 <- t.test(comment_length ~ `user_type`, data = new_df_length)

# Print t-test result
print(t_test_result3)


ggplot(new_df_length, aes(x = user_type, y = comment_length)) +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "point", shape = 20, size = 3, color = "red") +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "blue") +
  scale_x_discrete(labels = c("HUMAN" = "HUMAN", "AI" = "HUMANOID-AI")) +  # Manually set the labels
  labs(
    x = "User Type",
    y = "Text Length") +
  theme_minimal()



