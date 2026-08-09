# Emotional and behavioral asymmetries in user responses to virtual versus human influencers

## Overview
AI Virtual Influencers (AIVIs) are computer-generated personas that increasingly
occupy the same social media spaces as human influencers (HIs). Evidence on
whether they elicit comparable engagement is mixed.

We analyze 156,918 Instagram comments across 2,821 posts from 35 influencers
(23 AIVIs, 12 HIs) using a multi-signal computational framework combining
sentiment analysis, transformer-based emotion classification, LLM-derived
knowledge graphs, and machine learning.

HI-directed discourse shows cohesive semantic networks and more differentiated
emotional expression. AIVI-directed discourse occupies a more fragmented
semantic space, marked by neutral affect and limited emotional differentiation.
Notably, AIVIs receive fewer negative comments as well as fewer positive ones:
the asymmetry is reduced affective engagement in either direction rather than
increased hostility. A Multi-Signal Embedding Network classifies influencer type
from these discourse signatures (F1 = 0.89), outperforming a zero-shot LLM
baseline (F1 = 0.59).

These findings identify a boundary condition for the Computers Are Social Actors
(CASA) paradigm: human-like appearance can elicit social responses without
producing affective and semantic response patterns as rich as those elicited by
human influencers.

## Data Availability

Due to Instagram’s content sharing policies, we cannot share the raw dataset. 
However, a list of publicly available Instagram post URLs used in this study is provided to enable data reconstruction for reproducibility purposes.

## Running the code
`data_analysis.R` - 
This R script performs sentiment and popularity analysis on Instagram comment data, comparing interactions between users and two types of influencers: human (HI) and AI-based (AIVI).
Cleans comment text by removing usernames,
Applies sentiment analysis,
Computes comment length,
Performs statistical comparisons across groups (HI vs. AIVI) for:
Popularity,
Likes per post,
Likes per comment,
Sentiment scores,
Comment length, and
Visualizes results

Output: A CSV file with sentiment results and several plots highlighting group differences.

Note: `Data/sentiment_results_redacted.csv.zip` contains the per-comment
sentiment classifications together with post-level metadata. Comment text
and commenter identifiers are omitted, as Instagram's terms do not permit
their redistribution; post URLs are provided in
`Data/Instagram_Posts_URLs.txt`.

`EP_analysis.py` -
Analysis of Estimated Earnings per Post


`Text_to_Vector.py` -
Loads a JSON file (output_Final.json) containing user comments collected via the Meta Content Library API.
Starts processing at a given index (start_index = 150000) in case of a crash or to resume processing.
Uses OpenAI’s `text-embedding-3-small` model to generate embeddings from user comment text (posts.comments.text).
Stores the result in a new column: embedded.posts.comments.text.
Implements retrying failed embedding requests with backoff.
Periodically saves output to JSON files named by index (OutPutUntilXXXX.json), plus a final export.

`graph_analysis.R` - 
Reads subgraph edge lists and builds directed igraph graphs.
Computes multiple centrality metrics (Indegree, Outdegree, Closeness, etc.).
Applies log transformation to normalize skewed centrality distributions.
Performs Wilcoxon tests to compare groups (AIVI vs. HI).
Visualizes the results using ggplot2 boxplots with overlayed mean (blue dots) and median (red triangles).
Annotates significance (p-values) on each facet.

`model.R` - 
Train classification models (Logistic Regression and Neural Networks) to differentiate between user interactions with AI-based Virtual Influencers (AIVIs) and Human Influencers (HIs) based on comment data, metadata, sentiment, and vector embeddings.
Data Preparation & Sentiment Analysis:
Loads Excel/CSV comment data.
Cleans usernames and calculates sentiment using sentimentr.
Computes derived features like comment length and popularity.
Saves enriched data.
Statistical Testing (t-tests and Wilcoxon):
Tests for group differences in popularity, likes per post/comment, text length, and sentiment.
Visualizes these differences with ggplot2 boxplots and overlays mean/error bars.

Logistic Regression with caret:
Trains a logistic regression model on selected features (likes_count, popularity, sentiment).
Evaluates performance via confusion matrix, F1 score, and ROC/AUC.
Exploratory Model with BERT and TF-IDF:
Includes textEmbed for BERT embeddings and tm for TF-IDF vectorization.
Trains logistic regression using comment text.
Makes predictions and evaluates performance.

`Neural_Net.py` -
Loads a JSON file containing Instagram comment features and sentence embeddings.
Cleans and prepares the dataset:
Unpacks embedded vectors,
One-hot encodes sentiment,
Label-encodes used type (AIVI vs. HUMAN),
Trains a feed-forward neural network (MLPClassifier) to predict whether a comment was made in response to an AI or human influencer,
Evaluates the model using accuracy, classification report, confusion matrix, and ROC AUC.

`bert.py` - 
Extracts fixed-length text embeddings using a pre-trained BERT-base-uncased model and trains a logistic-regression classifier on them. 
Each text in X_text_column is tokenized and encoded with the BERT tokenizer, padded or truncated to a maximum length of 64 tokens. 
For each input, the [CLS] token vector (a 768-dimensional sentence representation from BERT’s last hidden layer) is extracted as its embedding.
These embeddings are stacked into a NumPy array (bert_features) that serves as input features for a 10-fold cross-validated logistic regression (LogisticRegressionCV) trained to predict the target labels y. 
The entire BERT model is used in inference mode (no fine-tuning), providing contextualized text representations for downstream classification.

`kg_construction.ipynb` - Builds the heterogeneous knowledge graph from user
comments. Each comment is processed with LangChain's LLMGraphTransformer using
OpenAI gpt-4o-mini to extract head-relation-tail triples, which are written to
a Neo4j database. Also contains the Cypher queries used to export the AIVI and
HI subgraph edge lists analyzed in `graph_analysis.R`.

`roberta-base-go_emotions.py` -
Fine-grained emotion analysis. Capture affective nuances beyond polarity-based sentiment.
This code runs a transformer-based multi-label emotion recognition model (SamLowe/roberta-base-go_emotions) fine-tuned on the GoEmotions dataset, which distinguishes 27 discrete emotions plus neutral (28 categories total). The model produces probability scores for each emotion, allowing multiple emotions to co-occur within a single comment.

`sampling_bot_or_not.py` -
Randomly sample comments directed at AIVIS and HIs.
Ensures each commenter appears only once.
Longest comment provides the richest text for manual evaluation.
140 AIVI + 140 HI, or fewer if you have fewer commenters.
Compute the proportion of bot-like commenters for AIVIs vs HIs.
Run a chi-square test.

`automated_activity_screen.py` - Assessment of potentially automated commenting
activity. Computes inter-rater agreement (Cohen's kappa) for the two-coder
profile assessment of 280 sampled commenters. Applies a coder-independent
behavioral screen measuring two repetition signatures: within-commenter
self-repetition and within-influencer cross-commenter template reuse.
Normalizes comment text by case folding and removal of punctuation, emoji and
diacritics, with collapsing of elongated characters. Computes rates per
influencer, compares groups using Mann-Whitney tests on the 35 influencer-level
rates, and reports Cliff's delta with bootstrap confidence intervals obtained by
resampling influencers within groups. Includes at-risk denominator sensitivity
analyses and leave-one-influencer-out stability checks.

Output: per-influencer repetition rates and group comparison statistics.

`sentiment_robustness.py` - Sensitivity analysis of the categorical sentiment
comparisons across four corpora: the original analysis corpus, a corpus
excluding comments with no text after normalization, a corpus excluding
comments flagged by either repetition criterion, and a corpus excluding both.
Reports prevalence, percentage-point differences, Cohen's h and odds ratios for
each condition.

Output: a comparison table across the four corpora.

`regenerate_polarity.R` - Regenerates the continuous sentimentr polarity score
for every comment. `data_analysis.R` persists only the discretized sign
(-1/0/1); this script recomputes and saves the underlying continuous values
using identical preprocessing (username removal only). Records the sentimentr,
R and lexicon versions used. Also decomposes exactly-zero-scoring comments via
`sentimentr::extract_sentiment_terms()` into "no polarized terms detected"
versus "polarized terms detected but summed to zero."

Note on versions: lexicon revisions between package releases shift continuous
polarity magnitudes while leaving the sign-based categorical classification
stable. The regenerated categorical counts match the published counts for
156,917 of 156,918 comments.

Output: `Data/polarity_scores.csv.zip` (continuous polarity, sign category, and
term-detection flag per comment, plus commenter id, influencer id, influencer
type, and comment length -- no raw comment text).

`polarity_threshold_sensitivity.py` - Sensitivity of the Positive/Neutral/
Negative categorization to the neutral-band width, using the continuous
polarity scores from `regenerate_polarity.R`. Reports the distribution of
polarity scores by influencer type, then reclassifies all comments under
neutral bands from 0 (the rule `data_analysis.R` actually implements) through
+/-0.25, reporting prevalence, the AIVI-HI neutrality gap, Cohen's h, odds
ratio, and Bonferroni-corrected chi-square significance at each band width.

Output: a comparison table across band widths and a plot of the neutrality
gap against band width.


## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to alon.bartal@biu.ac.il.


## Citing
If you find this paper useful for your research, please consider citing us:
```
@article{jagodnikAIVI,
  title={Emotional and behavioral asymmetries in user responses to virtual versus human influencers},
  author={Jagodnik, Kathleen M and Bartal, Alon},
  journal={Scientific Reports},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```


