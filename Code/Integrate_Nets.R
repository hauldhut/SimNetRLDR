library(RandomWalkRestartMH)
library(igraph)
library(foreach)
library(doParallel)

setwd("~/Manuscripts/125GNN4DR/Code")

DiSimNet1 <- read.delim("../Data/DiseaseSimNet_OMIM.txt",header = FALSE)
DiSimNet1.frame <- data.frame(DiSimNet1[[1]], DiSimNet1[[3]])
DiSimNet1.g <- graph.data.frame(d = DiSimNet1.frame, directed = FALSE)
DiSimNet1.weight = DiSimNet1[[2]]
E(DiSimNet1.g)$weight <- DiSimNet1.weight
cat(length(V(DiSimNet1.g)),length(E(DiSimNet1.g)),"\n")

DiSimNet2 <- read.delim("../Data/DiseaseSimNet_GeneNet.txt",header = FALSE)
DiSimNet2.frame <- data.frame(DiSimNet2[[1]], DiSimNet2[[3]])
DiSimNet2.g <- graph.data.frame(d = DiSimNet2.frame, directed = FALSE)
DiSimNet2.weight = DiSimNet2[[2]]
E(DiSimNet2.g)$weight <- DiSimNet2.weight
cat(length(V(DiSimNet2.g)),length(E(DiSimNet2.g)),"\n")

DiSimNet3 <- read.delim("../Data/DiseaseSimNet_HPO.txt",header = FALSE)
DiSimNet3.frame <- data.frame(DiSimNet3[[1]], DiSimNet3[[3]])
DiSimNet3.g <- graph.data.frame(d = DiSimNet3.frame, directed = FALSE)
DiSimNet3.weight = DiSimNet3[[2]]
E(DiSimNet3.g)$weight <- DiSimNet3.weight
cat(length(V(DiSimNet3.g)),length(E(DiSimNet3.g)),"\n")

#"per-edge average" integration
#Normalize
# hist(miRNA1.weight)#Already normalized
# hist(miRNA2.weight)#Already normalized

#Integrate 2
DiSimNet2I = merge(DiSimNet1, DiSimNet2, by=c("V1","V3"), all = TRUE)
DiSimNet2I[is.na(DiSimNet2I$V2.x),3] = 0
DiSimNet2I[is.na(DiSimNet2I$V2.y),4] = 0
DiSimNet2I$V2 = (DiSimNet2I$V2.x+DiSimNet2I$V2.y)/2

# Write V1, V2, V3 to tab-delimited file without headers
write.table(DiSimNet2I[,c("V1","V2","V3")], "../Data/DiseaseSimNet_MP.txt", 
            sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE)


#Integrate 3 
DiSimNet2I = merge(DiSimNet1, DiSimNet2, by=c("V1","V3"), all = TRUE)
DiSimNet3I = merge(DiSimNet2I, DiSimNet3, by=c("V1","V3"), all = TRUE)
DiSimNet3I[is.na(DiSimNet3I$V2.x),3] = 0
DiSimNet3I[is.na(DiSimNet3I$V2.y),4] = 0
DiSimNet3I[is.na(DiSimNet3I$V2),5] = 0

DiSimNet3I$W = (DiSimNet3I$V2.x+DiSimNet3I$V2.y+DiSimNet3I$V2)/3

# Write "V1","W","V3" to tab-delimited file without headers
write.table(DiSimNet3I[,c("V1","W","V3")], "../Data/DiseaseSimNet_MPH.txt", 
            sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE)

