#install.packages("car")
library(car)
library(dplyr)

library(tidyquant)
SPY <- c('SPY') %>% tq_get(get = "stock.prices", from = "2013-01-01", to = "2024-02-29")
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
head(SPY)
tail(SPY)
SPY_by_month <- group_by(SPY, month) %>% summarize(AVG_PRICE = mean(adjusted), AVG_VOLUME = mean(volume))
colnames(SPY_by_month)[1] <- "MONTH"
head(SPY_by_month)
tail(SPY_by_month)


data <- left_join(data, SPY_by_month, by = "MONTH")
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

model <- lm(AVG_PRICE~AVG_VOLUME+CPI+GDP+FEDRATE+JO+PCE+POP+UNRATE, data=data)
summary(model)

# Need to normalize the data.
