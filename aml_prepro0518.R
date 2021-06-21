#Preparation
raw_datapath <- "/media/dokada/HD-LBVU3/EEF/AML/data/"
out_path = "/home/dokada/work_dir/aml_prepro0518/"
if(!file.exists(out_path)){
  dir.create(out_path)
}
data_path = paste0(out_path,"trans_data/")
if(!file.exists(data_path)){
  dir.create(data_path)
}


#Calculation
library(flowCore)
files <- sort(list.files(raw_datapath))
n <- length(files)
all_marker2 <- c("FL1 Log","FL2 Log","FL3 Log","FL4 Log","FL5 Log")
all_protein2 <- c("IgG1-FITC","IgG1-PE","CD45-ECD","IgG1-PC5","IgG1-PC7")
for(i in 1:n){
  file1 <- files[i]
  fcs <- read.FCS(paste0(raw_datapath,file1),transformation=FALSE,dataset=1)
  expr <- asinh(fcs@exprs[,all_marker2]/5)
  colnames(expr) <- all_protein2
  write.csv(expr,file=paste0(data_path,file1,".csv"),quote=F,row.names=F)
  cat(i,"\n")
}

