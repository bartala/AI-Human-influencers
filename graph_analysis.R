library(readr)


HI <- AIVIHI[AIVIHI$user_type=='HUMAN',]
AIVI <- AIVIHI[AIVIHI$user_type=='AI',]


mean(HI$pot.comment.likes_count,na.rm = TRUE)
mean(AIVI$pot.comment.likes_count,na.rm = TRUE)

t.test(HI$posts.likes_count, AIVI$pot.comment.likes_count)



HI <- AIVI_HI[AIVI_HI$user_type=='HUMAN',]
AIVI <- AIVI_HI[AIVI_HI$user_type=='AI',]




#=============== Analyze sub-graphs =============================================

library(readr)
library(igraph)
library(dplyr)
library(ggplot2)
library(tidyr)

# Load the edgelists
neo4j_query_AI <- read_csv(".../AIVI_subgraph_edgelist.csv")
neo4j_query_HI <- read_csv(".../HI_subgraph_edgelist.csv")

# Build graphs
g_AI <- graph_from_data_frame(neo4j_query_AI, directed = TRUE)
g_HUMAN <- graph_from_data_frame(neo4j_query_HI, directed = TRUE)

# Function to compute centralities
compute_centralities <- function(g) {
  tibble(
    node = V(g)$name,
    #degree = degree(g, mode = "all"),
    Indegree = degree(g, mode = "in"),
    Outdegree = degree(g, mode = "out"),
    Closeness = closeness(g, normalized = TRUE),
    Betweenness = betweenness(g, normalized = TRUE),
    Eigenvector = eigen_centrality(g)$vector,
    Pagerank = page_rank(g)$vector,
    Authority = authority_score(g)$vector,
    Hub = hub_score(g)$vector,
    Coreness = coreness(g)
  )
}

# Compute and merge
df_AI <- compute_centralities(g_AI) %>% mutate(Group = "AIVI")
df_HUMAN <- compute_centralities(g_HUMAN) %>% mutate(Group = "HI")
df_all <- bind_rows(df_AI, df_HUMAN)

# Log-transform centralities (use log1p to avoid log(0))
centrality_cols <- c("Indegree", "Outdegree", "Closeness", "Betweenness", 
                     "Eigenvector", "Pagerank", "Authority", "Hub", "Coreness")

df_all_log <- df_all %>%
  mutate(across(all_of(centrality_cols), log1p))  # log1p = log(x + 1)

# Test for significance (Wilcoxon)
compare_centralities <- function(df, metric) {
  valid_rows <- !is.na(df[[metric]])
  df_valid <- df[valid_rows, ]
  if (length(unique(df_valid$Group)) == 2) {
    test <- wilcox.test(df_valid[[metric]] ~ df_valid$Group)
    return(data.frame(metric = metric, p_value = test$p.value))
  } else {
    return(data.frame(metric = metric, p_value = NA))
  }
}
tests <- lapply(centrality_cols, function(m) compare_centralities(df_all_log, m))
comparison_results <- bind_rows(tests) %>%
  mutate(p_label = paste0("p = ", signif(p_value, 3)))

# Prepare data for plotting
df_plot <- df_all_log %>%
  pivot_longer(cols = all_of(centrality_cols), names_to = "metric", values_to = "value") %>%
  left_join(comparison_results, by = "metric")


# Compute summary stats (mean and median)
summary_stats <- df_plot %>%
  group_by(Group, metric) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    .groups = "drop"
  )

# Melt the summary stats into long format
summary_long <- summary_stats %>%
  pivot_longer(cols = c(mean, median), names_to = "stat_type", values_to = "stat_value")

# Plot with boxplot + dots for mean (blue) and median (red)
ggplot(df_plot, aes(x = Group, y = value, fill = Group)) +
  geom_boxplot(outlier.size = 0.5, alpha = 0.5) +
  geom_point(data = summary_long,
             aes(x = Group, y = stat_value, color = stat_type, shape = stat_type),
             size = 2, position = position_dodge(width = 0.75)) +
  scale_color_manual(values = c(mean = "blue", median = "red"),
                     labels = c(mean = "Mean", median = "Median")) +
  scale_shape_manual(values = c(mean = 16, median = 17),
                     labels = c(mean = "Mean", median = "Median")) +
  facet_wrap(~ metric, scales = "free_y") +
  geom_text(data = df_plot %>% group_by(metric) %>% slice(1),
            aes(x = 1.5, y = Inf, label = p_label),
            vjust = 1.5, size = 3, inherit.aes = FALSE) +
  labs(title = "Log-Transformed Centrality Measures by Group (AI vs HUMAN)",
       y = "log(centrality + 1)", x = "",
       color = "Statistic", shape = "Statistic") +
  theme_minimal()
