# this function performs a population synthesis to generate synthetic vehicles 
# from the real processed data in Veh_features.csv

#Run A02 first to get Veh_features.csv

library("synthpop")

prject_path = '~/Documents/GitHub/data-driven-car-following/'

mydata = read.csv(paste(prject_path,"data/Veh_features.csv", sep="")) 

#vars <- c("sex", "age", "edu", "marital", "income", "ls", "wkabint")
#ods <- SD2011[, vars]
#head(ods)

my.seed <- 17914709
sds.default <- syn(mydata, seed = my.seed)

#sds.default

sds.parametric <- syn(mydata, method = "parametric", seed = my.seed)
#sds.parametric$method



