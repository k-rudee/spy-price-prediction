rm(list=ls())

#install.packages("car")
library(car)
library(dplyr)

library(tidyquant)
SPY <- c('SPY') %>% tq_get(get = "stock.prices", from = "2013-01-01", to = "2024-02-29")
head(SPY)
tail(SPY)

GLD <- c('GLD') %>% tq_get(get = "stock.prices", from = "2013-01-01", to = "2024-02-29")
head(SPY)
tail(SPY)

CPI <- read.csv("US CPI - Consumer Price Index.csv")
head(CPI)
tail(CPI)

GDP <- read.csv("US GDP.csv")
head(GDP)
tail(GDP)

FED_rate <- read.csv("US FED Funds Rate.csv")
head(FED_rate)
tail(FED_rate)

Job_Openings <- read.csv("US Job Openings Total Nonfarm.csv")
head(Job_Openings)
tail(Job_Openings)

PCE <- read.csv("US PCE - Personal Consumption Expenditure.csv")
head(PCE)
tail(PCE)

Pop <- read.csv("US Population.csv")
head(Pop)
tail(Pop)

U_rate <- read.csv("US Unemployment Rate.csv")
head(U_rate)
tail(U_rate)

data <- left_join(CPI, GDP, by = "DATE")
data <- left_join(data, FED_rate, by = "DATE")
data <- left_join(data, Job_Openings, by = "DATE")
data <- left_join(data, PCE, by = "DATE")
data <- left_join(data, Pop, by = "DATE")
data <- left_join(data, U_rate, by = "DATE")
data$Month <- format(as.Date(data$DATE), "%Y-%m")
head(data)
tail(data)
colnames(data)[2] <- "CPI"
colnames(data)[3] <- "GDP"
colnames(data)[4] <- "FEDRATE"
colnames(data)[5] <- "JO"
colnames(data)[6] <- "PCE"
colnames(data)[7] <- "POP"
colnames(data)[8] <- "UNRATE"
colnames(data)[9] <- "MONTH"

head(data)
tail(data)


SPY$month <- format(as.Date(SPY$date), "%Y-%m")
GLD$month <- format(as.Date(GLD$date), "%Y-%m")
head(SPY)
tail(SPY)
SPY_by_month <- group_by(SPY, month) %>% summarize(AVG_PRICE = mean(adjusted), AVG_VOLUME = mean(volume))
colnames(SPY_by_month)[1] <- "MONTH"
GLD_by_month <- group_by(GLD, month) %>% summarize(AVG_PRICE = mean(adjusted))
colnames(GLD_by_month)[1] <- "MONTH"
colnames(GLD_by_month)[2] <- "GLD_PRICE"
head(SPY_by_month)
head(GLD_by_month)
tail(GLD_by_month)


data <- left_join(data, SPY_by_month, by = "MONTH")
data <- left_join(data, GLD_by_month, by = "MONTH")
head(data)
tail(data)
data$GDP[1] <- mean(95.89, data$GDP[3])
data$GDP[2] <- mean(95.89, data$GDP[3])
data$GDP[118] <- data$GDP[117]
data$GDP[119] <- data$GDP[117]
data$GDP[120] <- data$GDP[117]

for (i in seq(3,114, by=3)){
  data$GDP[i+1] <- mean(data$GDP[i], data$GDP[i+3])
  data$GDP[i+2] <- mean(data$GDP[i], data$GDP[i+3])  
}

# Remove Feb as data was not fully available
data <- data[-121,]
head(data)
tail(data)


# Need to normalize the data as variables have very different ranges.
# Function to normalize data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to selected columns
data_normalized <- as.data.frame(lapply(data[, c('CPI', 'GDP', 'FEDRATE', 'JO', 'PCE', 'POP', 'UNRATE', 'AVG_VOLUME', 'GLD_PRICE')], normalize))
head(data_normalized)

data_normalized <- cbind(data[, c('DATE', 'MONTH', 'AVG_PRICE')], data_normalized)
head(data_normalized)
tail(data_normalized)


# Calculate correlation matrix for EDA
correlation_matrix <- cor(data_normalized[, c(-1,-2)])  # Exclude date and month columns from correlation calculation

# Print correlation matrix
print(correlation_matrix)

# Plot correlation matrix as a heatmap
library(ggplot2)
library(reshape2)

correlation_long <- melt(correlation_matrix)
ggplot(correlation_long, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Correlation Heatmap", x = "Variable 1", y = "Variable 2")

# Built the first linear regression model with all the 9 predicor variables
model <- lm(AVG_PRICE~AVG_VOLUME+CPI+GDP+FEDRATE+JO+PCE+POP+UNRATE+GLD_PRICE, data=data_normalized)
summary(model)

# The coefficients look very unusual compared to the correlation analysis.
# This indicates multicollinearity.
# Now perform VIF analysis to deal with multicollinearity. 

library(car)
# Run the vif function and show VIF values of the variables.
vif(model)
# VIF values of PCE, CPI, and GDP are too high. Try reducing number of variables by removing PCE.
model <- lm(AVG_PRICE~AVG_VOLUME+CPI+GDP+FEDRATE+JO+POP+UNRATE+GLD_PRICE, data=data_normalized)
summary(model)
vif(model)
# Coefficients looks less extreme, but it does not make sense that the coefficient of GDP is negative.
# Both the VIF values of CPI and GDP are still too high. Try remodeling with CPI removed.
model <- lm(AVG_PRICE~AVG_VOLUME+GDP+FEDRATE+JO+POP+UNRATE+GLD_PRICE, data=data_normalized)
summary(model)
vif(model)
# with p = 0.42, FEDRATE is too statistically insignificant. Try remodeling with FEDRATE removed.

model <- lm(AVG_PRICE~AVG_VOLUME+GDP+JO+POP+UNRATE+GLD_PRICE, data=data_normalized)
summary(model)
vif(model)
# Although VIF values of GDP, GLD_PRICE, and POP are still high (>5), they are much smaller from the original model.
# On the other hand, the coefficients look more reasonable with the correlation expectation with the SPY price.
# We will be using this model with 6 predictor variables to perform the train-validation-test split.

set.seed(68)
mask_train = sample(nrow(data_normalized), size = floor(nrow(data_normalized) * 0.6))
train = data_normalized[mask_train,] # training data set

# Using the remaining data for test and validation split

remaining = data_normalized[-mask_train, ]  # all rows except training

# Half of what's left for validation, half for test


mask_val = sample(nrow(remaining), size = floor(nrow(remaining)/2))

validate = remaining[mask_val,]  # validation data set
test = remaining[-mask_val, ] # test data set


model <- lm(AVG_PRICE~AVG_VOLUME+GDP+JO+POP+UNRATE+GLD_PRICE, data=train)
summary(model)
plot(model)

# R2 = 0.9684 is great. Residuals vs Fitted looks good. Q-Q Residuals look good.

# Now perform validation on the model
pred <- predict(model,validate[,c('AVG_VOLUME', 'GDP', 'JO', 'POP', 'UNRATE', 'GLD_PRICE')])
SST <- sum((validate$AVG_PRICE - mean(validate$AVG_PRICE))^2)
SSE <- sum((pred - validate$AVG_PRICE)^2)
# R-squared of the validation set
1 - SSE/SST
# R2 = 0.9635 is great. Roughly the same with that of the train data set.

pred <- predict(model,test[,c('AVG_VOLUME', 'GDP', 'JO', 'POP', 'UNRATE', 'GLD_PRICE')])
SST <- sum((test$AVG_PRICE - mean(test$AVG_PRICE))^2)
SSE <- sum((pred - test$AVG_PRICE)^2)
# R-squared of the test set
1 - SSE/SST 
# R2 = 0.9500 is great. This is just a little bit lower than that of the train and validation data sets.

# With this result, the linear regression model is good to be used to predict SPY price.


# Writing normalized data to csv files to create supporting visualizations in the Tableau Desktop software.
write.csv(data_normalized, "data_normalized.csv")
head(data)

data_normalized2 <- as.data.frame(lapply(data[, c('AVG_PRICE','CPI', 'GDP', 'FEDRATE', 'JO', 'PCE', 'POP', 'UNRATE', 'AVG_VOLUME', 'GLD_PRICE')], normalize))
data_normalized2 <- cbind(data[, c('DATE', 'MONTH')], data_normalized2)
# Writing normalized data with SPY Price normalized as well to csv files to create supporting visualizations in the Tableau Desktop software.
write.csv(data_normalized2, "data_normalized2.csv")

head(data_normalized2)

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, CPI) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, GDP) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, FEDRATE) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, JO) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, PCE) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, POP) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, UNRATE) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, AVG_VOLUME) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

data_normalized2 %>%
  tidyr::gather(key, value, AVG_PRICE, GLD_PRICE) %>%
  ggplot(aes(x=MONTH, y=value, colour=key, group=key)) +
  geom_line()

head(data)

pca <- prcomp(data[,c(-1,-9,-10)], scale.=TRUE)
summary(pca)

##########################
## PCA Visualizations  ###
##########################

# The following are useful visualizations when deciding how many principal components to choose.
# In this case, we are told to just use the first 4 principal components.

screeplot(pca, type="lines",col="blue")

# Calculate the variances and proportion of variances from the pca object

var <- pca$sdev^2
propvar <- var/sum(var)

# Plot the proportion of variances from PCA

plot(propvar, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")

# Plot the cumsum proportion of variances from PCA

cumsum(propvar)
plot(cumsum(propvar), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained",ylim = c(0,1), type = "b")

# The first 4 PCs take up 96.71% of the variance.

##########################
## Get first 4 PCs  ######
##########################

# Direct from prcomp output

PCs <- pca$x[,1:4]
attributes(pca$x)
pca$x
PCs


##########################
## Regress on first 4 PCs
##########################

# Build linear regression model with the first 4 principal components

SPY_pca <- cbind(PCs, data[,10]) #Create new data matrix with first 4 PCs and SPY Price

SPY_pca

as.data.frame(SPY_pca) #Shows why is it referencing V5

pca_model <- lm(V5~., data = as.data.frame(SPY_pca)) #Create regression model on new data matrix

summary(pca_model)

# While PC1, PC2, and PC4 are statistically significant, PC3 is not. How about a model with PC1, PC2, and PC4 only?
# Together, they account for 86.47% of variance
pca_model_2 <- lm(V5~PC1+PC2+PC4, data = as.data.frame(SPY_pca)) #Create regression model on new data matrix

summary(pca_model_2)
# Its R2 is almost the same as that of the model with the first four PCs.
# Both of them have marginally lower R2s than the straightforward model without using PCA.
# Using PCA does not result in better models. We do not continue exploring further as the non-PCA model
# as the non-PCA model already has good R2 resutls in the validation and test steps.


