#scRNA-seqのseparation scoreの3反復のscoreの相関をチェック(2021.6.21)
out_path <- "/home/dokada/work_dir/score_rep_check/"
if(!file.exists(out_path)) dir.create(out_path)
rep1 <- read.csv("/home/dokada/work_dir/uc_seed1/clf_scores_res.csv",header=T,row.names=1)
rep2 <- read.csv("/home/dokada/work_dir/uc_seed2/clf_scores_res.csv",header=T,row.names=1)
rep3 <- read.csv("/home/dokada/work_dir/uc_seed3/clf_scores_res.csv",header=T,row.names=1)
dat = data.frame(rep1$lambda,rep2$lambda,rep3$lambda)
colnames(dat) <- c("Rep1", "Rep2", "Rep3")

#cor plot
png(paste0(out_path, "corplot.png"))
plot(dat)
dev.off()

#cormat
cormat <- cor(dat, method="spearman")
write.csv(cormat, file=paste0(out_path,"cormat.csv"))
