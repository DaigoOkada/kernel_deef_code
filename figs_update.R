out_path <- "/home/dokada/work_dir/figs_update/"
#dir.create(out_path)

#Fig Supp
out_path <- "/home/dokada/work_dir/figs_update/"
X_iter0 <- read.csv("/home/dokada/work_dir/itn0518/X_iter0.csv",header=T,row.names=1)
X_iter1 <- read.csv("/home/dokada/work_dir/itn0518/X_iter1.csv",header=T,row.names=1)
X_iter2 <- read.csv("/home/dokada/work_dir/itn0518/X_iter2.csv",header=T,row.names=1)
dat1 <- as.data.frame(cbind(X_iter0[,1], X_iter1[,1], X_iter2[,1]))
colnames(dat1) <- c("Rep1", "Rep2", "Rep3")
dat2 <- as.data.frame(cbind(X_iter0[,2], X_iter1[,2], X_iter2[,2]))
colnames(dat2) <- c("Rep1", "Rep2", "Rep3")

png(paste0(out_path, "theta1_rep_plot.png"))
plot(dat1)
dev.off()

png(paste0(out_path, "theta2_rep_plot.png"))
plot(dat2)
dev.off()

#Fig2 QQ
out_path <- "/home/dokada/work_dir/figs_update/"
res <- read.csv("/home/dokada/work_dir/uc_seed1_0518/clf_scores_res.csv",header=T, row.names=1)
observed = sort(res[,"lambda"])
expected = sort(res[,"Null_lambda"])
log_exp_lambda = -log10(expected)
log_obs_lambda = -log10(observed)
png(paste0(out_path, "score_qq.png"))
plot(log_exp_lambda, log_obs_lambda,xlim=c(0,1),ylim=c(0,1))
abline(0, 1,col="red")
dev.off()






