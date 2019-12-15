#define functions
CVD_Age <- function(age, sex)
{
  if(sex == "Male") {
    return(CVD_Age_men(age))
  } else {
    return(CVD_Age_women(age))
  }
}

CVD_Age_SM <- function(age, sex)
{
  if(sex == "Male") {
    return(CVD_Age_men_SM(age))
  } else {
    return(CVD_Age_women_SM(age))
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

CVD_Age_men_SM <- function(age)
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
    return(7)    
  } else if (age < 55) {
    return(8)
  } else if (age < 60) {
    return(10)    
  } else if (age < 65) {
    return(11)    
  } else if (age < 70) {
    return(13)    
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

CVD_Age_women_SM <- function(age)
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

CVD_SBP_SM <- function(x, treated, sex)
{
  if(sex == "Male") {
    return(CVD_SBP_men_SM(x, treated))
  } else {
    return(CVD_SBP_women_SM(x, treated))
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

CVD_SBP_men_SM <- function(x, treated)
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

CVD_SBP_women_SM <- function(x, treated)
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
      return(8)
    }
  } else {
    if (x < 120 ) {
      return(-3)
    } else if (x < 130) {
      return(0)
    } else if (x < 140) {
      return(1)
    } else if (x <150) {
      return(3)
    } else if (x < 160) {
      return(4)
    } else {
      return(5)
    }
  }
}

CVD_Smoker <- function(smoker, sex)
{
  if(sex == "Male") {
    return(CVD_Smoker_men(smoker))
  } else {
    return(CVD_Smoker_women(smoker))
  }
}

CVD_Smoker_SM <- function(smoker, sex)
{
  return(CVD_Smoker_men(smoker))
}

CVD_Smoker_men <- function(smoker)
{
  if(smoker=="Current") {
    return(4)
  } else if (smoker == "Previous") {
    return(2) #WHAT TO DO with previous smokers?
  } else {
    return(0)
  }
}

CVD_Smoker_women <- function(smoker)
{
  if(smoker=="Current") {
    return(3)
  } else if (smoker == "Previous") {
    return(1.5) #WHAT TO DO with previous smokers?
  } else {
    return(0)
  }
}

CVD_Diabetic <- function(diabetic, sex)
{
  if(sex == "Male") {
    return(CVD_Diabetic_men(diabetic))
  } else {
    return(CVD_Diabetic_women(diabetic))
  }
}

CVD_Diabetic_SM <- function(diabetic, sex)
{
  if(sex == "Male") {
    return(CVD_Diabetic_men(diabetic)) #unchanged
  } else {
    return(CVD_Diabetic_women_SM(diabetic))
  }
}

CVD_Diabetic_men <- function(diabetic)
{
  if(diabetic == "Yes") {
    return(3)
  } else {
    return(0)
  }
}

CVD_Diabetic_women <- function(diabetic)
{
  if(diabetic == "Yes") {
    return(4)
  } else {
    return(0)
  }
}

CVD_Diabetic_women_SM <- function(diabetic)
{
  if(diabetic == "Yes") {
    return(5)
  } else {
    return(0)
  }
}

CVD_BMI <- function(bmi)
{
  if (bmi < 25) {
    return(0)
  } else if (bmi < 30) {
    return(1)
  } else {
    return(2)
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

Heart_Age_from_CVD_SM <- function(CVD, sex)
{
  if (sex == "Male") {
    return(Heart_Age_from_CVD_men_SM(CVD))
  } else {
    return(Heart_Age_from_CVD_women_SM(CVD))
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

Heart_Age_from_CVD_men_SM <- function(CVD)
{
  if (CVD < -1) {
    return(31 + (CVD+1)*2)
  } else if ( CVD == -1 ) {
    return(31)
  } else if ( CVD == 0 ) {
    return(33)
  } else if ( CVD == 1 ) {
    return(35)
  } else if ( CVD == 2 ) {
    return(37)
  } else if ( CVD == 3 ) {
    return(39)
  } else if ( CVD == 4 ) {
    return(41)
  } else if ( CVD == 5 ) {
    return(44)
  } else if ( CVD == 6 ) {
    return(46)
  } else if ( CVD == 7 ) {
    return(49)
  } else if ( CVD == 8 ) {
    return(52)
  } else if ( CVD == 9 ) {
    return(55)
  } else if ( CVD == 10 ) {
    return(58)
  } else if ( CVD == 11 ) {
    return(62)
  } else if ( CVD == 12 ) {
    return(65)
  } else if ( CVD == 13 ) {
    return(69)
  } else if ( CVD == 14 ) {
    return(73)
  } else if ( CVD == 15 ) {
    return(78)
  } else {
    return(78 + (CVD-15)*4)
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

Heart_Age_from_CVD_women_SM <- function(CVD)
{
  if (CVD < 0) {
    return(32 + (CVD)*2)
  } else if ( CVD == 0 ) {
    return(32)
  } else if ( CVD == 1 ) {
    return(34)
  } else if ( CVD == 2 ) {
    return(36)
  } else if ( CVD == 3 ) {
    return(38)
  } else if ( CVD == 4 ) {
    return(41)
  } else if ( CVD == 5 ) {
    return(43)
  } else if ( CVD == 6 ) {
    return(46)
  } else if ( CVD == 7 ) {
    return(48)
  } else if ( CVD == 8 ) {
    return(51)
  } else if ( CVD == 9 ) {
    return(54)
  } else if ( CVD == 10 ) {
    return(58)
  } else if ( CVD == 11 ) {
    return(61)
  } else if ( CVD == 12 ) {
    return(65)
  } else if ( CVD == 13 ) {
    return(69)
  } else if ( CVD == 14 ) {
    return(73)
  } else if ( CVD == 15 ) {
    return(77)
  } else {
    return(77 + (CVD-15)*4)
  } 
}
