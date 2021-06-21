#scRNA-seq
library(Seurat)
out_path <- "/home/dokada/work_dir/pbmc_prepro0518/"
out_path2 <- paste0(out_path,"prepro/")
dir.create(out_path)
dir.create(out_path2)


data_path <- "/home/dokada/work_dir/scrnaseq_bld0113/data_GSE125527/"
files <- sort(list.files(data_path))


#Processing
sample_names <- files
n <- length(sample_names)
data_list <- list()
for(i in 1:n){
    s <- sample_names[i]
    pbmc.data  <- read.table(paste0(data_path, s),header=T,row.names=1)
    pbmc <- CreateSeuratObject(counts =t(pbmc.data))
    pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT.")
    #VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
    pbmc_cellqc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 7000 & nCount_RNA < 70000 & percent.mt < 10)
    pbmc_cellqc_norm <- NormalizeData(pbmc_cellqc, normalization.method = "LogNormalize", scale.factor = 10000)
    mat_cellqc_norm = as.matrix(pbmc_cellqc_norm[["RNA"]]@data)
    data_list[[i]] <-  mat_cellqc_norm
    cat(i,"\n")
}

#QC for genes
pooled_reads_num <- rowSums(sapply(data_list,rowSums))
idx <- which(pooled_reads_num > 15000)
setwd(out_path2)
for(i in 1:n){
    tab <- t(data_list[[i]][idx,])
    write.csv(tab, file=paste0(sample_names[i], ".csv"),row.names=F)
}

#true label
true_label <- sapply(files, function(x){substr(strsplit(x,"_")[[1]][2],1,1)})
write.csv(true_label,file=paste0(out_path,"true_lab.csv"))

