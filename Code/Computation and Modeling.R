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

# Remove Feb as there is not data yet. When the data of GDP Q1 2024 is available, update the months of Q1 2024.
data <- data[-121,]
head(data)
tail(data)


# Need to normalize the data.
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


model <- lm(AVG_PRICE~AVG_VOLUME+CPI+GDP+FEDRATE+JO+PCE+POP+UNRATE+GLD_PRICE, data=data_normalized)
summary(model)

range(data$PCE)
range(data_normalized$CPI)


# Calculate correlation matrix
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

# Calculate correlation matrix
correlation_matrix <- cor(data[, c('DATE','MONTH')])  # Exclude date and month columns from correlation calculation

# Print correlation matrix
print(correlation_matrix)

# Create the scatter plot
plot(data_normalized$AVG_PRICE, data_normalized$CPI)

write.csv(data_normalized, "data_normalized.csv")
head(data)

data_normalized2 <- as.data.frame(lapply(data[, c('AVG_PRICE','CPI', 'GDP', 'FEDRATE', 'JO', 'PCE', 'POP', 'UNRATE', 'AVG_VOLUME', 'GLD_PRICE')], normalize))
data_normalized2 <- cbind(data[, c('DATE', 'MONTH')], data_normalized2)
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

# The first 4 PCs take up 96.71% of the variance

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
# Using PCA does not result in better models