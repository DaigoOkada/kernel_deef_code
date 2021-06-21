#ITN datasetにおけるDEEF(2021.6.21)
out_path <- "/home/dokada/work_dir/itn_original_deef_fin/"
dir.create(out_path)
#ITN data to P matrix
library(flowCore) #install from bioconducter
library(TDA) #install from CRAN
data_path <- "/home/dokada/work_dir/ITN_prepro0518/trans_data/"
files <- paste0(data_path, 1:15, ".csv")
annot <- read.csv("/home/dokada/work_dir/ITN_prepro0518/annot.csv",header=T,row.names=1)
n <- length(files)
expr_list <- list()
for(i in 1:n){
  samp <- read.csv(files[i],header=T)
  expr_list[[i]] <- asinh(as.matrix(samp))
}

#the num of marker(d)=2,the num of grids (m)=100
d <- ncol(samp)
m <- 10
min_list <- lapply(1:d,function(x){NULL})
max_list <- lapply(1:d,function(x){NULL})
alpha <- 0.15
for(i in 1:n){
  expr <- expr_list[[i]]
  for(j in 1:d){
    min_list[[j]] <- c(min_list[[j]],quantile(expr[,j],alpha))
    max_list[[j]] <- c(max_list[[j]],quantile(expr[,j],1-alpha))
  }
}

seq_list <- list()
for(j in 1:d){
  total_min <- min(min_list[[j]])
  total_max <- max(max_list[[j]])
  seq_list[[j]] <- seq(from=total_min,to=total_max,length=m)
}

#Generate Grid matrix
code <- "x_grid <- expand.grid("
for(i in 1:d){
  if(i != d) code <- paste0(code,"seq_list[[",i,"]],")
  if(i == d) code <- paste0(code,"seq_list[[",i,"]])")
}
eval(parse(text=code))

#knn estimate and generate P
P <- matrix(NA,n,m^d)
for(i in 1:n){
  expr <- expr_list[[i]]
  knni <- knnDE(expr, x_grid, k=100)
  P[i,] <- knni/sum(knni)
}


#DEEFを実行
#関数(package"deef"より）
DEEF <- function(disP, ip_mat=NULL){
  if(is.null(ip_mat)){
    ip_mat <- disP %*% t(disP)
  }
  theta_ip_est_mat <- log(ip_mat)/2

  #Eigenvalue decomposition
  eigen_out <- eigen(theta_ip_est_mat)
  eigen_value <- eigen_out[[1]]
  V <- eigen_out[[2]]
  Sigma <- diag(sqrt(abs(eigen_value)))
  Theta <- V %*% Sigma
  S <- diag(sign(eigen_value))

  #Calculate F'
  grid_num <- ncol(disP)
  sample_num <- nrow(disP)

  Psi <- matrix(NA,sample_num,grid_num)
  for(i in 1:sample_num){
    psi <- sum(sign(eigen_value) * Theta[i,]^2)
    Psi[i,] <- rep(psi,grid_num)
  }
  disP[disP==0] <- .Machine$double.xmin
  P_dash <- log(disP) + Psi
  Theta_dash <- cbind(Theta,rep(1,nrow(Theta)))
  F_dash <- MASS::ginv(Theta_dash) %*% P_dash

  #Output
  Theta <- Theta_dash[,-ncol(Theta_dash)]
  Cx <- F_dash[nrow(F_dash),]
  Fx <- F_dash[-nrow(F_dash),]
  result <- list(eigen_value,Theta,Cx,Fx)
  names(result) <- c("eigenvalue","Theta","Cx","Fx")
  return(result)
}

#DEEF
result <- DEEF(P)
eigen_value <- result$eigenvalue #all(sort(eigen_value,decreasing=T)==eigen_value) == TRUE
Theta <- result$Theta
all(Theta==Theta[,order(eigen_value,decreasing=T)])
Theta <- Theta[,order(eigen_value,decreasing=T)]
cols <- rep(NA,n)
true_labs = annot$GroupID
cols[true_labs=="Group 1"] = "red"
cols[true_labs=="Group 5"] = "blue"
cols[true_labs=="Group 6"] = "black"
png(paste0(out_path, "original_deef_and_kdeef.png"))
par(mfrow=c(1,2))
plot(Theta[,1], Theta[,2], col=cols, pch=16)


#origival deef only
png(paste0(out_path, "original_deef_only.png"))
plot(Theta[,1], Theta[,2], col=cols, pch=16, main="DEEF",cex.main=1.5,xlab="θ1",ylab="θ2",cex.lab=1.5)
dev.off()

