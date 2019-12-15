source("helpers_HeartAge.R")
set.seed(0)

#load data sample
#data <- readRDS("/n/groups/patel/uk_biobank/main_data_9512/ukb9512_10k_plyr.rds")
#Main file
data <- readRDS("/n/groups/patel/uk_biobank/main_data_9512/ukb9512_plyr.rds")
#Participant's ID: eid, Age when attended assessment centre: 21003, Sex: 31, BMI: 21001, SBP: 4080, SBP treated or not: 6153 and 6177, Smoker: 20116, Diabetic: 2443
toMatch <- c("f.6153", "f.6177")
matches <- unique(grep(paste(toMatch,collapse="|"), names(data), value=TRUE))
data_BP_treated <- data[,matches]
toMatch <- c("f.eid", "f.21003", "f.31", "f.21001", "f.4080", "f.6153", "f.6177", "f.20116", "f.2443")
matches <- unique(grep(paste(toMatch,collapse="|"), names(data), value=TRUE))
data_s <- data[,matches]
colSums(is.na(data_s)) #here: decide which matches to keep: select columns with limited NA%
tokeep <- c("f.eid", "f.21003.0.0", "f.31.0.0", "f.21001.0.0", "f.2443.0.0", "f.4080.0.0", "f.4080.0.1", "f.20116.0.0")
data_s <- data[,tokeep] 
#take mean of SBP measures, and remove the previous variables
data_s$SBP = rowMeans(data_s[c('f.4080.0.0', 'f.4080.0.1')], na.rm=TRUE)
data_s <- data_s[,which(!(names(data_s) %in% c('f.4080.0.0', 'f.4080.0.1')))]
data_s$BP_treated <- apply(data_BP_treated, 1, function(r) any(r %in% c("Blood pressure medication")))
names(data_s) <- c("eid", "Age", "Sex", "BMI", "Diabetic", "Smoker", "SBP", "BP_treated")

#biomarkers file: HDL: 30760, Cholesterol: 30690
data_biomarkers <- read.table('/n/groups/patel/uk_biobank/pheno_29961/ukb29961.csv', header = TRUE, sep = ",")
toMatch <- c("eid", "30760", "30690")
matches <- unique(grep(paste(toMatch,collapse="|"), names(data_biomarkers), value=TRUE))
datab_s <- data_biomarkers[,matches]
colSums(is.na(datab_s)) #select columns with limited NA%
tokeep <- c("eid", "X30690.0.0", "X30760.0.0")
datab_s <- data_biomarkers[,tokeep]
names(datab_s) <- c("eid", "Cholesterol", "HDL")

#merge the two datasets
data_targets <- merge(data_s, datab_s, by="eid")
#convert Cholesterol and HDL from mmol/L to mg/dL
data_targets$Cholesterol <- data_targets$Cholesterol*38.67
data_targets$HDL <- data_targets$HDL*38.67

#heart age
data_heart_age = data_targets[,which(names(data_targets) %in% c('eid', 'Age', 'Sex', 'BMI', 'SBP', 'BP_treated', 'Smoker', 'Diabetic', 'Cholesterol', 'HDL'))]
data_heart_age = data_heart_age[which(data_heart_age$Diabetic %in% c("No", "Yes") & data_heart_age$Smoker %in% c("Never", "Previous", "Current")),]
data_heart_age = data_heart_age[complete.cases(data_heart_age),]
#calculate the CVD points
data_heart_age$CVD <- mapply(CVD_Age, data_heart_age$Age, data_heart_age$Sex) + sapply(data_heart_age$HDL, CVD_HDL) + mapply(CVD_Cholesterol, data_heart_age$Cholesterol, data_heart_age$Sex) + mapply(CVD_SBP, data_heart_age$SBP, data_heart_age$BP_treated, data_heart_age$Sex) + mapply(CVD_Smoker, data_heart_age$Smoker, data_heart_age$Sex) + mapply(CVD_Diabetic, data_heart_age$Diabetic, data_heart_age$Sex)
#calculate heart age
data_heart_age$Heart_Age <- mapply(Heart_Age_from_CVD, data_heart_age$CVD, data_heart_age$Sex)
#delete other columns
data_heart_age <- data_heart_age[,c('eid', 'CVD', 'Heart_Age')]

#heart age SM
data_heart_age_SM = data_targets[,which(names(data_targets) %in% c('eid', 'Age', 'Sex', 'BMI', 'SBP', 'BP_treated', 'Smoker', 'Diabetic'))]
data_heart_age_SM = data_heart_age_SM[which(data_heart_age_SM$Diabetic %in% c("No", "Yes") & data_heart_age_SM$Smoker %in% c("Never", "Previous", "Current")),]
data_heart_age_SM = data_heart_age_SM[complete.cases(data_heart_age_SM),]
#calculate the CVD points
data_heart_age_SM$CVD_SM <- mapply(CVD_Age_SM, data_heart_age_SM$Age, data_heart_age_SM$Sex) + sapply(data_heart_age_SM$BMI, CVD_BMI) + mapply(CVD_SBP_SM, data_heart_age_SM$SBP, data_heart_age_SM$BP_treated, data_heart_age_SM$Sex) + mapply(CVD_Smoker_SM, data_heart_age_SM$Smoker, data_heart_age_SM$Sex) + mapply(CVD_Diabetic_SM, data_heart_age_SM$Diabetic, data_heart_age_SM$Sex)
#calculate heart age
data_heart_age_SM$Heart_Age_SM <- mapply(Heart_Age_from_CVD_SM, data_heart_age_SM$CVD_SM, data_heart_age_SM$Sex)
#delete other columns
data_heart_age_SM <- data_heart_age_SM[,c('eid', 'CVD_SM', 'Heart_Age_SM')]

#merge data
data_heart_ages <- merge(data_heart_age, data_heart_age_SM, by = 'eid', all = T)
data_targets <- merge(data_targets, data_heart_ages, by = 'eid', all = T)

#preprocess some variables
#Sex
data_targets$Sex <- abs(as.numeric(data_targets$Sex)-2)
#Age_group
data_targets$Age_group <- 0
data_targets$Age_group[which(data_targets$Age > 55)] <- 1
#Diabetic
data_targets$Diabetic <- as.numeric(data_targets$Diabetic)
data_targets$Diabetic[which(data_targets$Diabetic %in% c(1,2))] <- NA
data_targets$Diabetic[which(data_targets$Diabetic == 3)] <- 0
data_targets$Diabetic[which(data_targets$Diabetic == 4)] <- 1
#Smoker
data_targets$Smoker[which(data_targets$Smoker == "Prefer not to answer")] <- NA
data_targets$Smoker <- droplevels(data_targets$Smoker)

#save data_targets
write.csv(data_targets, "/n/groups/patel/Alan/Aging/ECG/data/data_targets.csv", row.names=F)

#load ecg features and merge them with the dataset
data_targets <- read.table('../data/data_targets.csv', header = TRUE, sep = ",")
data_ecg <- read.table('../data/preprocessed_ECG_features.csv', header = TRUE, sep = ",")
data <- merge(data_targets, data_ecg, by = "eid")
#get rid of low quality ECG rows
#only keep rows for which '12-lead ECG measuring method' is 0 (Direct Entry, as opposed to not performed)
data <- data[which(data$X12.lead.ECG.measuring.method==0),]
#remove rows based on 'Suspicious flag for 12-lead ECG'. Only non NA value is 0, approximately 5% of the cases for which ECG info is available. Removing these samples
data <- data[-which(data$Suspicious.flag.for.12.lead.ECG==0),]
#remove '12-lead ECG measuring method' and 'Suspicious flag for 12-lead ECG' columns
data <- data[,-which(names(data) %in% c("X12.lead.ECG.measuring.method", "Suspicious.flag.for.12.lead.ECG"))]

#print missing values
print("Printing number of missing values for each variable:")
print(colSums(is.na(data)))

#print summary target variables
print("Printing summary statistics for each variable:")
print(summary(data))

#plot correlations between quantitative targets
print(cor(data[complete.cases(data),-which(names(data) %in% c("eid", "Smoker", "BP_treated"))]))

#shuffle
data <- data[sample(nrow(data)),]
#save file
write.csv(data, "/n/groups/patel/Alan/Aging/ECG/data/data_targets_and_features.csv", row.names=F)

#data <- read.table("/n/groups/patel/Alan/Aging/ECG/data/data_targets.csv", header = TRUE, sep = ",")


