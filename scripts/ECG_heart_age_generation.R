#load data sample
#data <- readRDS("/n/groups/patel/uk_biobank/main_data_9512/ukb9512_10k_plyr.rds")
#Main file
data <- readRDS("/n/groups/patel/uk_biobank/main_data_9512/ukb9512_plyr.rds")
#Participant's ID: eid, Age when attended assessment centre: 21003, Sex: 31, SBP: 4080, SBP treated or not: 6153 and 6177, Smoking: 20116, Diabetic: 2443
toMatch <- c("f.6153", "f.6177")
matches <- unique(grep(paste(toMatch,collapse="|"), names(data), value=TRUE))
data_BP_treated <- data[,matches]
toMatch <- c("f.eid", "f.21003", "f.31", "f.4080", "f.6153", "f.6177", "f.20116", "f.2443")
matches <- unique(grep(paste(toMatch,collapse="|"), names(data), value=TRUE))
data_s <- data[,matches]
colSums(is.na(data_s)) #here: decide which matches to keep: select columns with limited NA%
tokeep <- c("f.eid", "f.21003.0.0", "f.31.0.0", "f.2443.0.0", "f.4080.0.0", "f.4080.0.1", "f.20116.0.0")
data_s <- data[,tokeep]
#take mean of SBP measures, and remove the previous variables
data_s$SBP = rowMeans(data_s[c('f.4080.0.0', 'f.4080.0.1')], na.rm=TRUE)
data_s <- data_s[,which(!(names(data_s) %in% c('f.4080.0.0', 'f.4080.0.1')))]
data_s$BP_treated <- apply(data_s, 1, function(r) any(r %in% c("Blood pressure medication")))
names(data_s) <- c("eid", "Age", "Sex", "Diabetes", "Smoking", "SBP", "BP_treated")

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
data_heart_age <- merge(data_s, datab_s, by="eid")
data_heart_age = data_heart_age[complete.cases(data_heart_age),]
#convert Cholesterol and HDL from mmol/L to mg/dL
data_heart_age$Cholesterol <- data_heart_age$Cholesterol*38.67
data_heart_age$HDL <- data_heart_age$HDL*38.67

#define functions
CVD_Age <- function(age, sex)
{
  if(sex == "Male") {
    return(CVD_Age_men(age))
  } else {
    return(CVD_Age_women(age))
  }
}

CVD_Age_men <- function(age)
{
  if (age < 30) {
    return(NA)
  } else if (age < 35) {
    return(0)
  } else if (age < 40) {
    return(2)
  } else if (age < 45) {
    return(5)    
  } else if (age < 50) {
    return(6)    
  } else if (age < 55) {
    return(8)    
  } else if (age < 60) {
    return(10)    
  } else if (age < 65) {
    return(11)    
  } else if (age < 70) {
    return(12)    
  } else if (age < 75) {
    return(14)    
  } else {
    return(15)
  }
}

CVD_Age_women <- function(age)
{
  if (age < 30) {
    return(NA)
  } else if (age < 35) {
    return(0)
  } else if (age < 40) {
    return(2)
  } else if (age < 45) {
    return(4)    
  } else if (age < 50) {
    return(5)    
  } else if (age < 55) {
    return(7)    
  } else if (age < 60) {
    return(8)    
  } else if (age < 65) {
    return(9)    
  } else if (age < 70) {
    return(10)    
  } else if (age < 75) {
    return(11)    
  } else {
    return(12)
  }
}

CVD_HDL <- function(x)
{
  if (x > 59 ) {
    return(-2)
  } else if (x > 49) {
    return(-1)
  } else if (x > 44) {
    return(0)
  } else if (x > 34) {
    return(1)
  } else {
    return(2)
  }
}

CVD_Cholesterol <- function(x, sex)
{
  if(sex == "Male") {
    return(CVD_Cholesterol_men(x))
  } else {
    return(CVD_Cholesterol_women(x))
  }
}

CVD_Cholesterol_men <- function(x)
{
  if (x < 160 ) {
    return(0)
  } else if (x < 200) {
    return(1)
  } else if (x < 240) {
    return(2)
  } else if (x < 280) {
    return(3)
  } else {
    return(4)
  }
}

CVD_Cholesterol_women <- function(x)
{
  if (x < 160 ) {
    return(0)
  } else if (x < 200) {
    return(1)
  } else if (x < 240) {
    return(3)
  } else if (x < 280) {
    return(4)
  } else {
    return(5)
  }
}

CVD_SBP <- function(x, treated, sex)
{
  if(sex == "Male") {
    return(CVD_SBP_men(x, treated))
  } else {
    return(CVD_SBP_women(x, treated))
  }
}

CVD_SBP_men <- function(x, treated)
{
  if (treated) {
    if (x < 120 ) {
      return(0)
    } else if (x < 130) {
      return(2)
    } else if (x < 140) {
      return(3)
    } else if (x < 160) {
      return(4)
    } else {
      return(5)
    }
  } else {
    if (x < 120 ) {
      return(-2)
    } else if (x < 130) {
      return(0)
    } else if (x < 140) {
      return(1)
    } else if (x < 160) {
      return(2)
    } else {
      return(3)
    } 
  }
}

CVD_SBP_women <- function(x, treated)
{
  if (treated) {
    if (x < 120 ) {
      return(-1)
    } else if (x < 130) {
      return(2)
    } else if (x < 140) {
      return(3)
    } else if (x < 150) {
      return(5)
    } else if (x < 160) {
      return(6)
    } else {
      return(7)
    }
  } else {
    if (x < 120 ) {
      return(-3)
    } else if (x < 130) {
      return(0)
    } else if (x < 140) {
      return(1)
    } else if (x <150) {
      return(2)
    } else if (x < 160) {
      return(4)
    } else {
      return(5)
    } 
  }
}

CVD_Smoking <- function(smoking, sex)
{
  if(sex == "Male") {
    return(CVD_Smoking_men(smoking))
  } else {
    return(CVD_Smoking_women(smoking))
  }
}

CVD_Smoking_men <- function(smoking)
{
  if(smoking=="Current") {
    return(4)
  } else if (smoking == "Previous") {
    return(2) #WHAT TO DO with previous smokers?
  } else {
    return(0)
  }
}

CVD_Smoking_women <- function(smoking)
{
  if(smoking=="Current") {
    return(3)
  } else if (smoking == "Previous") {
    return(1.5) #WHAT TO DO with previous smokers?
  } else {
    return(0)
  }
}

CVD_Diabetes <- function(diabetes, sex)
{
  if(sex == "Male") {
    return(CVD_Diabetes_men(diabetes))
  } else {
    return(CVD_Diabetes_women(diabetes))
  }
}

CVD_Diabetes_men <- function(diabetes)
{
  if(diabetes == "Yes") {
    return(3)
  } else {
    return(0)
  }
}

CVD_Diabetes_women <- function(diabetes)
{
  if(diabetes == "Yes") {
    return(4)
  } else {
    return(0)
  }
}

Heart_Age_from_CVD <- function(CVD, sex)
{
  if (sex == "Male") {
    return(Heart_Age_from_CVD_men(CVD))
  } else {
    return(Heart_Age_from_CVD_women(CVD))
  }
}

Heart_Age_from_CVD_men <- function(CVD)
{
  if (CVD < 0) {
    return(30 + (CVD)*2)
  } else if ( CVD == 0 ) {
    return(30)
  } else if ( CVD == 1 ) {
    return(32)
  } else if ( CVD == 2 ) {
    return(34)
  } else if ( CVD == 3 ) {
    return(36)
  } else if ( CVD == 4 ) {
    return(38)
  } else if ( CVD == 5 ) {
    return(40)
  } else if ( CVD == 6 ) {
    return(42)
  } else if ( CVD == 7 ) {
    return(45)
  } else if ( CVD == 8 ) {
    return(48)
  } else if ( CVD == 9 ) {
    return(51)
  } else if ( CVD == 10 ) {
    return(54)
  } else if ( CVD == 11 ) {
    return(57)
  } else if ( CVD == 12 ) {
    return(60)
  } else if ( CVD == 13 ) {
    return(64)
  } else if ( CVD == 14 ) {
    return(68)
  } else if ( CVD == 15 ) {
    return(72)
  } else if ( CVD == 16 ) {
    return(76)
  } else {
    return(76 + (CVD-16)*4)
  } 
}

Heart_Age_from_CVD_women <- function(CVD)
{
  if (CVD < 1) {
    return(31 + (CVD-1)*3)
  } else if ( CVD == 1 ) {
    return(31)
  } else if ( CVD == 2 ) {
    return(34)
  } else if ( CVD == 3 ) {
    return(36)
  } else if ( CVD == 4 ) {
    return(39)
  } else if ( CVD == 5 ) {
    return(42)
  } else if ( CVD == 6 ) {
    return(45)
  } else if ( CVD == 7 ) {
    return(48)
  } else if ( CVD == 8 ) {
    return(51)
  } else if ( CVD == 9 ) {
    return(55)
  } else if ( CVD == 10 ) {
    return(59)
  } else if ( CVD == 11 ) {
    return(64)
  } else if ( CVD == 12 ) {
    return(68)
  } else if ( CVD == 13 ) {
    return(73)
  } else if ( CVD == 14 ) {
    return(79)
  } else {
    return(79 + (CVD-14)*5)
  } 
}

#calculate the CVD points
data_heart_age$CVD <- mapply(CVD_Age, data_heart_age$Age, data_heart_age$Sex) + sapply(data_heart_age$HDL, CVD_HDL) + mapply(CVD_Cholesterol, data_heart_age$Cholesterol, data_heart_age$Sex) + mapply(CVD_SBP, data_heart_age$SBP, data_heart_age$BP_treated, data_heart_age$Sex) + mapply(CVD_Smoking, data_heart_age$Smoking, data_heart_age$Sex) + mapply(CVD_Diabetes, data_heart_age$Diabetes, data_heart_age$Sex)
#calculate heart age
data_heart_age$Heart_Age <- mapply(Heart_Age_from_CVD, data_heart_age$CVD, data_heart_age$Sex)
data_heart_age <- data_heart_age[sample(nrow(data_heart_age)),]
#save file
write.csv(data_heart_age, "/n/groups/patel/Alan/Aging/ECG/data/data_heart_age.csv", row.names=F)

#DO IT ON PYTHON
# #take subset for which ECG data is available, on train, val and test.
# data_heart_age = read.table("/n/groups/patel/Alan/Aging/ECG/data/data_heart_age.csv", sep=",", header=T)
# folds = c('train', 'val', 'test')
# for (k in seq(length(folds)))
# {
#   fold = folds[k]
#   data_targets <- read.table(paste('/n/groups/patel/Alan/Aging/ECG/data/demo_', fold, '.csv', sep=''), sep=",", header=T)
#   names(data_targets) <- c("eid", "Age", "Sex")
#   data_targets_fold = data_heart_age[which(data_heart_age$eid %in% data_targets$eid),]
#   write.csv(data_targets_fold, paste('/n/groups/patel/Alan/Aging/ECG/data/data_targets_', fold, '.csv', sep=''), row.names=F)
# }
# 

