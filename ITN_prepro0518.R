#Preparation
out_path <- "/home/dokada/work_dir/ITN_prepro0518/"

if(!file.exists(out_path)) dir.create(out_path)
data_path1 <- paste0(out_path,"trans_data/")
if(!file.exists(data_path1)) dir.create(data_path1)


#library
library(flowStats)
data(ITN)
n <- length(ITN)

#Calculation
annot <- NULL
for(i in 1:n){
  samp <- ITN[[i]]
  expr <- asinh(samp@exprs[,c("CD8", "CD69", "CD4", "CD3", "HLADr")]/5)
  annot <- rbind(annot,ITN[i]@phenoData@data)
  write.csv(expr, file=paste0(data_path1, i,".csv"), row.names=F)
}
write.csv(annot, file=paste0(out_path, "annot.csv"), row.names=T)
