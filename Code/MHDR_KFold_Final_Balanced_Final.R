Method = "MH123_Balanced"#MH123/MH12/MH13/MH23/H1/H2/H3/H1_Balanced/H3_Balanced/MH13_Balanced/MH123_Balanced
gamma = 0.5

start_time <- Sys.time()

# Load required library
library(dplyr)
library(RandomWalkRestartMH)
library(igraph)

#need to install foreach and doParallel packages for this code to run
library(foreach)
library(doParallel)

setwd("~/Manuscripts/125GNN4DR/Code")

#DiseaseSimNet_OMIM.txt
#DiseaseSimNet_HPO.sif
#DiseaseSimNet_GeneNet.txt

DiSimNet1 <- read.delim("../Data/DiseaseSimNet_OMIM.txt",header = FALSE)
DiSimNet1.frame <- data.frame(DiSimNet1[[1]], DiSimNet1[[3]])
DiSimNet1.g <- graph.data.frame(d = DiSimNet1.frame, directed = FALSE)
DiSimNet1.weight = DiSimNet1[[2]]
E(DiSimNet1.g)$weight <- DiSimNet1.weight

DiSimNet2 <- read.delim("../Data/DiseaseSimNet_HPO.txt",header = FALSE)
DiSimNet2.frame <- data.frame(DiSimNet2[[1]], DiSimNet2[[3]])
DiSimNet2.g <- graph.data.frame(d = DiSimNet2.frame, directed = FALSE)
DiSimNet2.weight = DiSimNet2[[2]]
E(DiSimNet2.g)$weight <- DiSimNet2.weight

DiSimNet3 <- read.delim("../Data/DiseaseSimNet_GeneNet.txt",header = FALSE)
DiSimNet3.frame <- data.frame(DiSimNet3[[1]], DiSimNet3[[3]])
DiSimNet3.g <- graph.data.frame(d = DiSimNet3.frame, directed = FALSE)
DiSimNet3.weight = DiSimNet3[[2]]
E(DiSimNet3.g)$weight <- DiSimNet3.weight

if(Method == "MH123_Balanced"){
  disease_MultiplexObject <- create.multiplex(list(DiSimNet1.g,DiSimNet2.g,DiSimNet3.g),Layers_Name = c("DiSimNet1","DiSimNet2","DiSimNet3"))  
  tau1 = 1
  tau2 = 1
  tau3 = 1
  tau <- c(tau1, tau2, tau3)
}else if(Method == "MH12"){
  disease_MultiplexObject <- create.multiplex(list(DiSimNet1.g,DiSimNet2.g),Layers_Name = c("DiSimNet1","DiSimNet2"))
  tau1 = 1
  tau2 = 1
  tau <- c(tau1, tau2)
}else if(Method == "MH13_Balanced"){
  disease_MultiplexObject <- create.multiplex(list(DiSimNet1.g,DiSimNet3.g),Layers_Name = c("DiSimNet1","DiSimNet3"))
  tau1 = 1
  tau2 = 1
  tau <- c(tau1, tau2)
}else if(Method == "MH23"){
  disease_MultiplexObject <- create.multiplex(list(DiSimNet2.g,DiSimNet3.g),Layers_Name = c("DiSimNet2","DiSimNet3"))
  tau1 = 1
  tau2 = 1
  tau <- c(tau1, tau2)
}else if(Method == "H1_Balanced"){
  disease_MultiplexObject <- create.multiplex(list(DiSimNet1.g),Layers_Name = c("DiSimNet1"))
  tau <- c(1)
}else if(Method == "H2"){
  disease_MultiplexObject <- create.multiplex(list(DiSimNet2.g),Layers_Name = c("DiSimNet2"))
  tau <- c(1)
}else if(Method == "H3_Balanced"){
  disease_MultiplexObject <- create.multiplex(list(DiSimNet3.g),Layers_Name = c("DiSimNet3"))
  tau <- c(1)
}
# }else{ #H1_Balanced/H3_Balanced/MH13_Balanced/MH123_Balanced
#   disease_MultiplexObject <- create.multiplex(list(DiSimNet1.g,DiSimNet2.g,DiSimNet3.g),Layers_Name = c("DiSimNet1","DiSimNet2","DiSimNet3"))  
#   tau1 = 1
#   tau2 = 1
#   tau3 = 1
#   tau <- c(tau1, tau2, tau3)
# }

#DrugSimNet_PREDICT.txt
#DrugSimNet_CHEM.txt
DrugSimNet = "CHEM"
if(DrugSimNet == "PREDICT"){
  DrSimNet <- read.delim("../Data/DrugSimNet_PREDICT.txt",header = FALSE)
}else{
  DrSimNet <- read.delim("../Data/DrugSimNet_CHEM.txt",header = FALSE)  
}

DrSimNet.frame <- data.frame(DrSimNet[[1]], DrSimNet[[3]])
DrSimNet.weight = DrSimNet[[2]]

DrSimNet.g <- graph.data.frame(d = DrSimNet.frame, directed = FALSE)
E(DrSimNet.g)$weight <- DrSimNet.weight

drug_MultiplexObject <- create.multiplex(list(DrSimNet.g),
                                         Layers_Name = c("DrSimNet"))

#Add DiDrRelation
DiDr.frame <- read.csv("../Data/Drug2Disease_PREDICT_BinaryInteraction.csv", header = TRUE)
DiDr.frame <- DiDr.frame[which(DiDr.frame$disease %in% disease_MultiplexObject$Pool_of_Nodes),]
DiDr.frame <- DiDr.frame[which(DiDr.frame$drug %in% drug_MultiplexObject$Pool_of_Nodes),]

#add func for RWR on multiplex-heter nw
do_something <- function(disease_MultiplexObject,drug_MultiplexObject,
                         DiDrRelation,SeedDisease, seeddrug, prd_diseases) {
  
  #Create multiplex-heterosgenous nw
  DiDrRelation_disease <- DiDrRelation[which(DiDrRelation$disease %in% disease_MultiplexObject$Pool_of_Nodes),]
  
  #Create multiplex-heterosgenous nw
  disease_drug_Net <- create.multiplexHet(disease_MultiplexObject, drug_MultiplexObject, 
                                          DiDrRelation_disease)
  
  disease_drug_Net_TranMatrix <- compute.transition.matrix(disease_drug_Net)
  
  #compute 
  Ranking_Results <- Random.Walk.Restart.MultiplexHet(disease_drug_Net_TranMatrix,
                                                      disease_drug_Net,SeedDisease,
                                                      seeddrug, r = gamma)
  
  #create labels for ranking results
  tf = Ranking_Results$RWRMH_Multiplex1
  
  tf$labels <- ifelse(tf$NodeNames %in% prd_diseases, 1, 0)
  
  # Select all nodes with label=1 and equal number of random nodes with label=0
  label_1_indices <- which(tf$labels == 1)
  label_0_indices <- which(tf$labels == 0)
  n_label_1 <- length(label_1_indices)
  if (n_label_1 > 0 && length(label_0_indices) >= n_label_1) {
    sampled_label_0_indices <- sample(label_0_indices, n_label_1)
    selected_indices <- c(label_1_indices, sampled_label_0_indices)
    tf <- tf[selected_indices, ]
  }
  
  # calculating AUC
  resultspred = prediction(tf$Score, tf$labels)
  
  auc.perf = performance(resultspred, measure = "auc")
  auroc = auc.perf@y.values[[1]]
  
  aupr.perf = performance(resultspred, measure = "aucpr")
  auprc = aupr.perf@y.values[[1]]
  
  return(list(auroc, auprc,data.frame(Scores=tf$Score, Labels=tf$labels)))
}

#count disease for each drug
sub_sum <- aggregate(disease~drug, data=DiDr.frame, FUN=function(x) c(count=length(x)))

#extract drug with only k or more diseases
k=10
sub_sum <- sub_sum[which(sub_sum$disease>=k),]
sub_sum$drug_no <- c(1:length(sub_sum$drug))

#extract DiDr.frame with only disease from sub_sum
DiDr.frame1 <- DiDr.frame[which(DiDr.frame$drug %in% sub_sum$drug),]
rownames(DiDr.frame1) <- NULL #reset frame index

#func to assign k groups for each set of drug-disease, 
#as well as increment group no. for each group (k=3)
assign_group_no <- function(sub_sum,DiDr.frame1,k) {
  
  #set an empty data frame for a new DiDr.frame
  mylist.names <- c("drug","disease", "drug_no","group_no")
  DiDr.frame2 <- sapply(mylist.names,function(x) NULL)
  
  for (j in 1:length(sub_sum$drug)) {
    
    count = sub_sum$disease[[j]]
    
    if(count<k) next
    
    set_no = floor(count/k)
    
    print(paste(k,count, set_no))
    
    group_vec = vector()
    for(gi  in 1:(k-1)){
      group_vec = c(group_vec,rep(gi,set_no))
    }
    group_vec = c(group_vec,rep(k,count-set_no*(k-1)))
    
    subset <- DiDr.frame[which(DiDr.frame$drug==sub_sum$drug[[j]]),]
    subset$drug_no <- rep(j,count)
    subset$group <- group_vec
    
    DiDr.frame2 <- rbind(DiDr.frame2,subset)
  }
  return(DiDr.frame2)
}

#assign each group of n/k disease-drug with an group id
out <- assign_group_no(sub_sum,DiDr.frame1,k)
out$group <- (out$drug_no-1)*k+out$group
DiDr.frame2 <- out

#set an empty data frame for a new DiDr.frame
auc_results <- sapply(c("disease","group","auc"),function(x) NULL)

for (i in 1:max(DiDr.frame2$group)) {
  seeddrug = unique(DiDr.frame2$drug[which(DiDr.frame2$group==i)])
  auc_results$drug[i] <- seeddrug
  auc_results$group[i] <- i
}

#set up paralell processing (adjust the no_cores as per running system)
no_cores <- 6
cl <- makeCluster(no_cores)
registerDoParallel(cl)

#loop through
res <- foreach(i = 1:max(DiDr.frame2$group), .combine = rbind) %dopar% {
  
  library(RandomWalkRestartMH)
  library(igraph)
  library(ROCR)
  
  prd_diseases = DiDr.frame2$disease[which(DiDr.frame2$group==i)]
  seeddrug = unique(DiDr.frame2$drug[which(DiDr.frame2$group==i)])
  
  disease_relation = DiDr.frame2[which(DiDr.frame2$drug==seeddrug),]
  SeedDisease = disease_relation$disease[-c(which(disease_relation$disease %in% prd_diseases))]
  
  # get bipartite graph without prd_diseases - disease linkages
  DiDrRelation <- DiDr.frame2[-with(DiDr.frame2, which(disease %in% prd_diseases & drug %in% seeddrug)),][1:2]
  
  res <- do_something(disease_MultiplexObject,drug_MultiplexObject,
                      DiDrRelation,SeedDisease, seeddrug, prd_diseases)
  res <- append(res, seeddrug)
}

dim(res)
stopCluster(cl)

df.res = data.frame(trial = c(1:nrow(res)),auc = unlist(res[,1]),aupr = unlist(res[,2]))
df.res.final = merge(df.res, DiDr.frame2[c("drug","group")], by.x="trial", by.y="group", all.x = TRUE)
df.res.final = unique(df.res.final)

Result_byTrialFile = paste0("../Results/",Method,"_byTrial_",DrugSimNet,"_ROC_KFold10.csv")
cat(Result_byTrialFile,"\n")
write.csv(df.res.final,Result_byTrialFile, row.names = FALSE, quote = FALSE)

aucavgbyTrial = round(mean(df.res.final$auc),3)
aucavgbyTrial.sd = round(sd(df.res.final$auc),3)

aupravgbyTrial = round(mean(df.res.final$aupr),3)
aupravgbyTrial.sd = round(sd(df.res.final$aupr),3)

cat("Method=",Method,'DrugSimNet=',DrugSimNet,"\n")
print("By Trial")
cat('aucavgbyTrial=',aucavgbyTrial,"(+-",aucavgbyTrial.sd,")\n")
cat('aupravgbyTrial=',aupravgbyTrial,"(+-",aupravgbyTrial.sd,")\n")

##### Summarize by Drug/Disease


# Group by drug and calculate mean and standard deviation for auc and aupr
df.res.final.byD <- df.res.final %>%
  group_by(drug) %>%
  summarise(
    auc.mean = mean(auc),
    auc.std = sd(auc),
    aupr.mean = mean(aupr),
    aupr.std = sd(aupr)
  )

# Save the resulting data frame to a CSV file
Result_byDFile = paste0("../Results/",Method,"_byD_",DrugSimNet,"_ROC_KFold10.csv")
write.csv(df.res.final.byD, Result_byDFile, row.names = FALSE, quote = FALSE)

# Calculate mean and standard deviation for auc.mean and aupr.mean
auc_mean_overall <- round(mean(df.res.final.byD$auc.mean), 3)
auc_std_overall <- round(sd(df.res.final.byD$auc.mean), 3)
aupr_mean_overall <- round(mean(df.res.final.byD$aupr.mean), 3)
aupr_std_overall <- round(sd(df.res.final.byD$aupr.std), 3)

# Create a data frame to store the results
overall_stats <- data.frame(
  metric = c("auc.mean", "aupr.mean"),
  mean = c(auc_mean_overall, aupr_mean_overall),
  std = c(auc_std_overall, aupr_std_overall)
)

# Print the results
print("By Disease/Drug")
print(overall_stats)

##### Summarize All --> only one value for each metric 
res.final = NULL
for(i in 1:nrow(res)){
  # print(res[i,3][[1]])
  res.final = rbind(res.final, res[i,3][[1]])
}

Result_by_Score_n_LabelFile = paste0("../Results/",Method,"_byTrial_",DrugSimNet,"_Score_n_Label_KFold10.csv")
cat(Result_by_Score_n_LabelFile,"\n")
write.csv(res.final,Result_by_Score_n_LabelFile, row.names = FALSE, quote = FALSE)

library(ROCR)
resultspred = prediction(res.final$Scores, res.final$Labels)
auc.perf = performance(resultspred, measure = "auc")
aucavgbyAll = round(auc.perf@y.values[[1]],3)

aupr.perf = performance(resultspred, measure = "aucpr")
aupravgbyAll = round(aupr.perf@y.values[[1]],3)

print("Summarize All --> only one value for each metric")
cat("Method=",Method,'DrugSimNet=',DrugSimNet,'aucavgbyAll=',aucavgbyAll,"\n")
cat('aucavgbyAll=',aucavgbyAll,"(+-",overall_stats$std[1],")\n")
cat('aupravgbyAll=',aupravgbyAll,"(+-",overall_stats$std[2],")\n")

end_time <- Sys.time()
timediff = end_time - start_time
print(timediff)