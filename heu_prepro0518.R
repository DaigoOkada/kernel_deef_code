#Preparation
raw_datapath <- "/media/dokada/HD-LBVU3/EEF/HEU/data/"
out_path = "/home/dokada/work_dir/heu_prepro0518/"
if(!file.exists(out_path)){
  dir.create(out_path)
}
data_path = paste0(out_path,"trans_data/")
if(!file.exists(data_path)){
  dir.create(data_path)
}

#calculation
library(flowCore)
files <- sort(list.files(raw_datapath))
n <- length(files)
all_marker <- c("FITC-A","PE-A","PerCP-Cy5-5-A","PE-Cy7-A","APC-A","APC-Cy7-A","Pacific Blue-A","Alex 700-A")
all_protein <- c("IFNa","CD123","MHCII","CD14","CD11c","IL6","IL-12","TNF-a")
for(i in 1:n){
  file1 <- files[i]
  fcs <- read.FCS(paste0(raw_datapath,file1),transformation=FALSE)
  expr <- fcs@exprs[,all_marker]
  expr <- asinh(expr/5)
  colnames(expr) <- all_protein
  write.csv(expr,file=paste0(data_path,file1,".csv"),quote=F,row.names=F)
  cat(i,"\n")
}


#annotation file
annot <- read.csv("/media/dokada/HD-LBVU3/EEF/HEU/attachments/HEUvsUE.csv")
write.csv(annot,file=paste0(out_path,"annot.csv"),quote=F)




