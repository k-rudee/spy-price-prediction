
# clear environment
rm(list = ls())
set.seed(1)

# load packages
library(randomForest)
library(tree)
library(caret)
library(dplyr)

# data load
library(tidyquant)
spy_data <- c('SPY') %>% tq_get(get = "stock.prices", from = "2013-01-01", to = "2024-02-29")
crude_oil <- read.csv("crude_oil.csv", header = TRUE)
gold <- read.csv("gold.csv", header = TRUE)
natural_gas <- read.csv("natural_gas.csv", header = TRUE)


# stock_market_data <- read.csv("Stock Market Dataset.csv", header = TRUE)

# merge data together to a master dataframe
data <- merge(spy_data, crude_oil, by.x = "date", by.y = "date",  all.x = TRUE, all.y = FALSE)
data <- merge(data, gold, by.x = "date", by.y = "date",  all.x = TRUE, all.y = FALSE)
data <- merge(data, natural_gas, by.x = "date", by.y = "date",  all.x = TRUE, all.y = FALSE)


# data clean to update column names and remove na values
data_clean <- data %>% select("date","volume","adjusted","crude_oil_price","gold_price","natural_gas_price")
data_clean <- drop_na(data_clean)
summary(data_clean)
nrow(data_clean)

# split into train validate test
ss <- sample(1:3,size=nrow(data_clean),replace=TRUE,prob=c(0.6,0.2,0.2))
train <- data_clean[ss==1,]
val <- data_clean[ss==2,]
test <- data_clean[ss==3,]


# regression tree model
spy_tree_model <- tree(adjusted ~ crude_oil_price + gold_price + natural_gas_price, data = train)

# summary of the model
summary(spy_tree_model)

# show the tree
spy_tree_model$frame
plot(spy_tree_model)
text(spy_tree_model)

# check performance of the model on validation data
spy_tree_predict <- predict(spy_tree_model, data = test[,4:6])
RSS <- sum((spy_tree_predict - test[,3])^2)
TSS <- sum((test[,3] - mean(test[,3]))^2)
R2 <- 1 - RSS/TSS
R2

# random forest model
spy_rf_model <- randomForest(adjusted ~ crude_oil_price + gold_price + natural_gas_price, data = train, importance = TRUE)
spy_rf_predict <- predict(spy_rf_model, data = test[,4:6])
RSS <- sum((spy_rf_predict - test[,3])^2)
R2 <- 1 - RSS/TSS
R2


