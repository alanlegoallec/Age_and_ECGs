
#load data sample
#data <- readRDS("/n/groups/patel/uk_biobank/main_data_9512/ukb9512_10k_plyr.rds")
#load data
data <- readRDS("/n/groups/patel/uk_biobank/main_data_9512/ukb9512_plyr.rds")

#Participant's ID: eid, Age when attended assessment centre: 21003, Sex: 31, Ethnicity: 21000 
toMatch <- c("f.eid", "f.21003", "f.31", "f.21000")
matches <- unique(grep(paste(toMatch,collapse="|"), names(data), value=TRUE))
data_s <- data[,matches]
#here: decide which matches to keep
tokeep <- c("f.eid", "f.21003.0.0", "f.31.0.0", "f.21000.0.0")
data_s <- data[,tokeep]
names(data_s) <- c("eid", "Age", "Sex", "Ethnicity")
data_s=data_s[complete.cases(data_s),]
write.csv(data_s, "/n/groups/patel/Alan/Aging/ECG/data/preprocessed_demographics.csv")
