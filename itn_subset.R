#ITN dataset, cellular subset analysis(2021.6.2)
#Refer to:
#https://www.bioconductor.org/packages/release/bioc/manuals/flowStats/man/flowStats.pdfのtutorialに沿う
#https://www.bioconductor.org/packages/release/bioc/vignettes/flowWorkspace/inst/doc/flowWorkspace-Introduction.html
#https://github.com/RGLab/flowCore/issues/167
#https://bioconductor.riken.jp/packages/3.9/workflows/vignettes/highthroughputassays/inst/doc/high-throughput-assays.htmlS

#path
out_path <- "/home/dokada/work_dir/itn_subset/"
dir.create(out_path)


##
library(flowCore)
library(flowStats)
library(flowWorkspace)
library(ggcyto)
data(ITN)
require(scales)
gs <- GatingSet(ITN)
trans.func <- asinh
inv.func <- sinh
trans.obj <- trans_new("myAsinh", trans.func, inv.func)
transList <- transformerList(colnames(ITN)[3:7], trans.obj)
gs <- transform(gs, transList)




#visualization
#DEEF
annot <- read.csv("/home/dokada/work_dir/ITN_prepro0518/annot.csv",header=T,row.names=1)
n <- nrow(annot) 
cols <- rep(NA,n)
true_labs = annot$GroupID #gs@data@phenoData@data
cols[true_labs=="Group 1"] = "red"
cols[true_labs=="Group 5"] = "blue"
cols[true_labs=="Group 6"] = "black"


#T cell gate and CD4/CD8 gate
lg <- lymphGate(gs_cyto_data(gs), channels=c("CD3", "SSC"),preselection="CD4", filterId="TCells", scale=2.5)
gs_pop_add(gs, lg)
recompute(gs)
qgate <- quadrantGate(gs_pop_get_data(gs, "TCells"), stains=c("CD4", "CD8"), filterId="CD4CD8", sd=3)
gs_pop_add(gs, qgate, parent = "TCells")
recompute(gs)
cs_t <- gs_pop_get_data(gs, "/TCells")
gs <- normalize(gs, populations=c("CD4+CD8+", "CD4+CD8-", "CD4-CD8+", "CD4-CD8-"), dims=c("CD4", "CD8"), minCountThreshold = 50)
cs_kil <- gs_pop_get_data(gs, "/TCells/CD4-CD8+")
cs_hel <- gs_pop_get_data(gs, "/TCells/CD4+CD8-")
cs_t2 <- gs_pop_get_data(gs, "/TCells")
kil_frac <- rep(NA, n)
hel_frac <- rep(NA, n)
t_frac <- rep(NA, n)
t_frac2 <- rep(NA, n)
for(i in 1:n){
  kil_frac[i] <- nrow(cs_kil[[i]])
  hel_frac[i] <- nrow(cs_hel[[i]])
  t_frac[i] <- nrow(cs_t[[i]])
  t_frac2[i] <- nrow(cs_t2[[i]])
}

#all(t_frac==t_frac2) TRUE

#plot
png(paste0(out_path, "cell_subset_plot.png"), height=640, width=640)
plot(hel_frac, kil_frac, col=cols, pch=16, cex.lab=1.4, xlab="CD4+ T cell [cells]", ylab="CD8+ T cell [cells]", cex=2) 
dev.off()

#CD4/CD8 gating
png(paste0(out_path, "gating_plot.png"))
ggcyto(gs_pop_get_data(gs, "TCells"), aes(x=CD4, y=CD8)) +
  geom_hex(bins=32) +
  geom_gate(gs_pop_get_gate(gs, "CD4-CD8-")) +
  geom_gate(gs_pop_get_gate(gs, "CD4-CD8+")) +
  geom_gate(gs_pop_get_gate(gs, "CD4+CD8-")) +
  geom_gate(gs_pop_get_gate(gs, "CD4+CD8+"))
dev.off()
