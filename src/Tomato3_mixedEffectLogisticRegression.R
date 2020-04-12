library(lme4)
library(rms)
setwd("D:/OneDrive/Projects/Overtaking/Experiment_Tomato3/Tomato3_src")
data = read.csv("overtaking.csv")


m <- glmer(overtaking ~ fspd_s + e + fmean + (fspd_s + e + fmean| subj), 
           data = data, family = "binomial", control = glmerControl(optimizer = "bobyqa"),
           nAGQ = 1)
summary(m)



m4 <- glm(overtaking ~ fspd_s : e : fmean, data = data, family = "binomial")
summary(m4)
BIC(m4)


m4 <- lrm(overtaking ~ lspd + fspd_s + fmean, data = data)
print(m4)
