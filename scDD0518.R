#scDD analysis

#out path
library(scDD)
out_path <- "/home/dokada/work_dir/scDD0518/"
dir.create(out_path)


#UC datasetで実施
set.seed(1000)
data_path <- "/home/dokada/work_dir/pbmc_prepro0518/prepro/"
files <- sort(list.files(data_path))
labs <- read.csv("/home/dokada/work_dir/pbmc_prepro0518/true_lab.csv",header=T,row.names=1)
U_files <- files[labs=="U"]
C_files <- files[labs=="C"]

#Set n_pic
n_vec <- rep(NA, length(files))
for(i in 1:length(files)){
  dat <- read.csv(paste0(data_path, files[i]),header=T)
  n_vec[i] <- nrow(dat)
}
n_pic <- min(n_vec) #672

#U mat
U_mat <- NULL
for(i in 1:length(U_files)){
  dat <- read.csv(paste0(data_path, U_files[i]),header=T)
  n <- nrow(dat)
  U_mat <- rbind(U_mat, dat[sample(1:n, n_pic, replace=F),])
}

#C mat
C_mat <- NULL
for(i in 1:length(C_files)){
  dat <- read.csv(paste0(data_path, C_files[i]),header=T)
  n <- nrow(dat)
  C_mat <- rbind(C_mat, dat[sample(1:n, n_pic, replace=F),])
}

#Create dataset
condition1 <- c(rep(1,nrow(C_mat)),rep(2,nrow(U_mat)))
UC_mat <- cbind(t(C_mat), t(U_mat))
scDat <- SingleCellExperiment::SingleCellExperiment(assays=list(normcounts=UC_mat))
scDat@colData@listData$condition <-  condition1
scDat <- scDD(scDat, testZeroes=FALSE) #prior_paramはdefaultが使用される
RES <- results(scDat)
png(paste0(out_path, "scDD_phist.png"))
hist(RES$nonzero.pvalue, xlab="P values",cex.lab=1.5,main="scDD P values")
dev.off()

#Manova lambdaとの比較
manova_res <- read.csv("/home/dokada/work_dir/uc_seed1_0518/clf_scores_res.csv",header=T, row.names=1)
x <- manova_res$lambda
y <- RES$nonzero.pvalue
y[y==0] <- min(y[y!=0]) * 0.1
png(paste0(out_path, "scDD_manova_coplot.png"))
plot(-log10(x), -log10(y), xlab="-log10(lambda)", ylab="-log10(scDD Pval)",cex.lab=1.5,cex.main=2)
dev.off()

#Create dataset for null distribution
condition2 <- sample(c(rep(1,nrow(C_mat)),rep(2,nrow(U_mat))))
scDat_rand <- SingleCellExperiment::SingleCellExperiment(assays=list(normcounts=UC_mat))
scDat_rand@colData@listData$condition <-  condition2
scDat_rand  <- scDD(scDat_rand, testZeroes=FALSE) #prior_paramはdefaultが使用される
RES_rand  <- results(scDat_rand)
png(paste0(out_path, "scDD_rand_hist.png"))
hist(RES_rand$nonzero.pvalue, xlab="P values",cex.lab=1.5,main="scDD P values")
dev.off()


#QQ plot
y <- RES$nonzero.pvalue
y_null <- RES_rand$nonzero.pvalue
y[y==0] <- min(y[y!=0]) * 0.1
y_null[y_null==0] <- min(y_null[y_null!=0]) * 0.1
png(paste0(out_path, "scDD_qq.png"))
plot(sort(-log10(y_null)), sort(-log10(y)), xlab="-log10(Expected P)", ylab="-log10(Observed P)",cex.lab=1.5,cex.main=2)
abline(0, 1,col="red")
dev.off()
