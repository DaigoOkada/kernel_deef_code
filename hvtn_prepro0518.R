#Preparation
raw_datapath <- "/media/dokada/HD-LBVU3/EEF/HVTN/data/"
out_path = "/home/dokada/work_dir/hvtn_prepro0518/"
if(!file.exists(out_path)){
  dir.create(out_path)
}
data_path = paste0(out_path,"trans_data/")
if(!file.exists(data_path)){
  dir.create(data_path)
}


#analysis
library(flowCore)
files <- sort(list.files(raw_datapath))
n <- length(files)
all_marker <- c("<FITC-A>","<Alexa 680-A>","<APC-A>","<PE Cy7-A>","<PE Cy55-A>","<PE Tx RD-A>","<PE Green laser-A>")
all_protein <- c("CD4","TNFa","IL4","IFNg","CD8","CD3","IL2")
for(i in 1:n){
  file1 <- files[i]
  fcs <- read.FCS(paste0(raw_datapath,file1),transformation=FALSE)
  expr <- fcs@exprs[,all_marker]
  expr <- asinh(expr/5)
  colnames(expr) <- all_protein 
  write.csv(expr,file=paste0(data_path,file1,".csv"),quote=F,row.names=F)
  cat(i,"\n")
}


#Annotation files
annot <- read.csv("/media/dokada/HD-LBVU3/EEF/HVTN/attachments/Challenge3Metadata.csv")
annot_stm <- annot[(annot$Sample.Treatment=="GAG-1-PTEG")|(annot$Sample.Treatment=="ENV-1-PTEG"),]
write.csv(annot_stm,file=paste0(out_path,"annot_stm.csv"),quote=F)


