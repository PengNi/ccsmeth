## #!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
library(ggplot2)
library(grid)
library(gridExtra)
library(reshape2)
library(scales)

rmetfile1 = args[1]  # rmet file 1, not necessaryly be a bs report
rmetfile2 = args[2] # rmet file 2, not necessayly be a nanopore file
rmetfile1name = args[3]
rmetfile2name = args[4]
# hx1_label = "HX1.BJXWZ.10x"
rmetfile2label = args[5]

is_plot = args[6]  # "yes" or "no"
resultdir = args[7]
resultname = args[8]

f1_header = TRUE
f2_header = FALSE
str2bool = function(input_str)
{
    if(input_str == "yes"){
        input_str = TRUE
    }else{
        input_str = FALSE
    }
    return(input_str)
}
if(length(args)>=9){
  f1_header = args[9] # "yes" or "no"
  f1_header = str2bool(f1_header)
}
if(length(args)>=10){
  f2_header = args[10] # "yes" or "no"
  f2_header = str2bool(f2_header)
}

midfix="" # mainly for bisulfite data type/source, e.g. ENCFF835NTC
if(length(args)>=11){
  midfix=args[11]
}

allsites = "no"
if(length(args)>=12){
  allsites = args[12]
}

# functions ==================
ratio_stats <- function(values, lb=0, ub=1, xbin=0.1){
  sranges <- seq(lb, ub, xbin)
  bincounts <- vector('numeric', length=length(sranges)-1)
  bincounts[1] <- sum(sranges[1] <= values & values <= sranges[2])
  for(i in 2:(length(sranges)-1)){
    bincounts[i] <- sum(sranges[i] < values & values <= sranges[i+1])
  }
  x <- data.frame(idx=c(1:length(bincounts)), 
                  ranges=sranges[1:(length(sranges)-1)], 
                  counts=bincounts, 
                  ratio=bincounts/length(values))
}

classify_one_type <- function(rmet, sranges){
  for(i in 2:(length(sranges))){
    if(rmet<=sranges[i]){return(i-1)}
  }
}

classify_rtype <- function(Rmet, sranges){
  return(sapply(Rmet, classify_one_type, sranges))
}

generate_rtype_matrix <- function(Rmet1, Rmet2, rescale_num=10){
  scale_range <- 1 / rescale_num
  sranges <- seq(0, 1, scale_range)
  
  res_mat <- matrix(data=0, nrow=length(sranges)-1, ncol=length(sranges)-1, 
                    dimnames = list(sranges[2:length(sranges)], 
                                    sranges[2:length(sranges)]))
  # res_mat <- matrix(data=0, nrow=length(sranges)+1, ncol=length(sranges)+1, 
  #                   dimnames = list(c(sranges, '1+'), c(sranges, '1+')))
  rtype1 = classify_rtype(Rmet1, sranges)
  rtype2 = classify_rtype(Rmet2, sranges)
  for(i in 1:length(rtype1)){
    tmp = res_mat[rtype1[i], rtype2[i]]
    res_mat[rtype1[i], rtype2[i]] = tmp + 1
  }
  res_mat
}

heatplot <- function(heat_mat, xlab="", ylab="", color.low="#ffffff", color.high="#e41a1c", 
                     labs=TRUE, digits=2, labs.size=4, 
                     font.size=14, readable=FALSE ) {
  sim.df <- as.data.frame(heat_mat)
  if(readable == TRUE) {
    ##rownames(sim.df) <- TERM2NAME.DO(rownames(sim.df))
    ##colnames(sim.df) <- TERM2NAME.DO(colnames(sim.df))
  }
  rn <- row.names(sim.df)
  
  sim.df <- cbind(ID=rownames(sim.df), sim.df)
  sim.df <- melt(sim.df)
  
  sim.df[,1] <- factor(sim.df[,1], levels=rev(rn))
  if (labs == TRUE) {
    ## lbs <- c(apply(round(sim, digits), 2, as.character))
    sim.df$label <- as.character(round(sim.df$value, digits))
  }
  variable <- ID <- value <- label <- NULL ## to satisfy codetools
  if (labs == TRUE)
    p <- ggplot(sim.df, aes(variable, ID, fill=value, label=label))
  else
    p <- ggplot(sim.df, aes(variable, ID, fill=value))
  
  p <- p + geom_tile(color="black")+
    scale_fill_gradient(low=color.low, high=color.high) +
    scale_x_discrete(expand=c(0,0)) +
    scale_y_discrete(expand=c(0,0))+
    theme(axis.ticks=element_blank())
  if (labs == TRUE)
    p <- p+geom_text(size=labs.size)
  # p <- p+theme_dose(font.size)
  p <- p + theme(axis.text.x=element_text(size=10, hjust=0, angle=-90)) +
    theme(axis.text.y=element_text(size=10, hjust=0))
  p <- p+theme(legend.title=element_blank())
  p <- p+theme(axis.title = element_text(size = 15))
  ##geom_point(aes(size=value))
  p <- p+xlab(xlab)+ylab(ylab)
  
  if (readable == TRUE) {
    p <- p + theme(axis.text.y = element_text(hjust=1))
  }
  p <- p + theme(axis.text.x = element_text(vjust=0.5))
  return(p)
}

disaplay_corr_eq <- function(rmet_bis, rmet_nan, lb, ub){
  rmet_bis_lu <- rmet_bis[rmet_bis >= lb & rmet_bis <= ub]
  rmet_nan_lu <- rmet_nan[rmet_bis >= lb & rmet_bis <= ub]
  message(sprintf("rmet corr, [%.3f, %.3f] (len: %d), pearson: %f, spearman: %f", lb, ub,
                  length(rmet_bis_lu), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='pearson'), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='spearman')))
}
disaplay_corr_neq <- function(rmet_bis, rmet_nan, lb, ub){
  rmet_bis_lu <- rmet_bis[rmet_bis > lb & rmet_bis < ub]
  rmet_nan_lu <- rmet_nan[rmet_bis > lb & rmet_bis < ub]
  message(sprintf("rmet corr, (%.3f, %.3f) (len: %d), pearson: %f, spearman: %f", lb, ub,
                  length(rmet_bis_lu), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='pearson'), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='spearman')))
}
disaplay_corr_leq <- function(rmet_bis, rmet_nan, lb, ub){
  rmet_bis_lu <- rmet_bis[rmet_bis >= lb & rmet_bis < ub]
  rmet_nan_lu <- rmet_nan[rmet_bis >= lb & rmet_bis < ub]
  message(sprintf("rmet corr, [%.3f, %.3f) (len: %d), pearson: %f, spearman: %f", lb, ub,
                  length(rmet_bis_lu), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='pearson'), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='spearman')))
}
disaplay_corr_ueq <- function(rmet_bis, rmet_nan, lb, ub){
  rmet_bis_lu <- rmet_bis[rmet_bis > lb & rmet_bis <= ub]
  rmet_nan_lu <- rmet_nan[rmet_bis > lb & rmet_bis <= ub]
  message(sprintf("rmet corr, (%.3f, %.3f] (len: %d), pearson: %f, spearman: %f", lb, ub,
                  length(rmet_bis_lu), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='pearson'), 
                  cor(rmet_bis_lu, rmet_nan_lu, method='spearman')))
}


# reading =======================================
datacov = rmetfile2label
bisulfite_data <- read.table(rmetfile1, 
                             header = f1_header, sep = "\t", stringsAsFactors = F)
nanopore_data <- read.table(rmetfile2, 
                            header = f2_header, sep = "\t", stringsAsFactors = F)
# name_sample1 = 'bisulfite'
# name_sample2 = 'deepsignal.th0'
name_sample1 = rmetfile1name
name_sample2 = rmetfile2name

# fill up rmet1 info ==
if (ncol(bisulfite_data)==5){
  bisulfite_data$cRmet <- bisulfite_data$Rmet
} else if(substr(name_sample1, 1, 10)=='nanopolish'){
  colnames(bisulfite_data) <- c("chromosome", "pos", "pos_in_strand", "num_motifs_in_group", 
                                "coverage", "met", "Rmet", "group_sequence")
} else if(substr(name_sample1, 1, 14)=='deepsignal_bed'){
  colnames(bisulfite_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3",
                                "na4", "na5", "coverage", "rpercent")
  bisulfite_data$Rmet <- bisulfite_data$rpercent / 100
} else if(substr(name_sample1, 1, 10)=='deepsignal' & ncol(bisulfite_data)==11){
  colnames(bisulfite_data) <- c("chromosome", "pos", "strand", "pos_in_strand", 
                                "prob0", "prob1", "met", "unmet", "coverage", 
                                "Rmet", "kmer")
  bisulfite_data$cRmet <- bisulfite_data$Rmet
} else if(substr(name_sample1, 1, 9)=='megalodon'){
  colnames(bisulfite_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3", 
                                "na4", "na5", "coverage", "rpercent")
  bisulfite_data$Rmet <- bisulfite_data$rpercent / 100
} else if(substr(name_sample1, 1, 5)=='tombo'){
  colnames(bisulfite_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3",
                                "na4", "na5", "coverage", "rpercent")
  bisulfite_data$Rmet <- bisulfite_data$rpercent / 100
} else if(substr(name_sample1, 1, 13)=='bisulfite_bed'){
  colnames(bisulfite_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3",
                                "na4", "na5", "coverage", "rpercent")
  bisulfite_data$Rmet <- bisulfite_data$rpercent / 100
  bisulfite_data$cRmet <- bisulfite_data$Rmet
} else if(substr(name_sample1, 1, 7)=='methccs'){
  colnames(bisulfite_data) <- c("chromosome", "pos", "strand",
                                "prob0", "prob1", "met", "unmet", "coverage",
                                "Rmet", "kmer")
  bisulfite_data$cRmet <- bisulfite_data$Rmet
}

# fill up rmet2 info ==
if (ncol(nanopore_data)==5){
  nanopore_data$cRmet <- nanopore_data$Rmet
} else if(substr(name_sample2, 1, 10)=='nanopolish'){
  colnames(nanopore_data) <- c("chromosome", "pos", "pos_in_strand", "num_motifs_in_group", 
                               "coverage", "met", "Rmet", "group_sequence")
} else if(substr(name_sample2, 1, 14)=='deepsignal_bed'){
  colnames(nanopore_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3",
                                "na4", "na5", "coverage", "rpercent")
  nanopore_data$Rmet <- nanopore_data$rpercent / 100
} else if(substr(name_sample2, 1, 10)=='deepsignal' & ncol(nanopore_data)==11){
  colnames(nanopore_data) <- c("chromosome", "pos", "strand", "pos_in_strand", 
                                "prob0", "prob1", "met", "unmet", "coverage", 
                                "Rmet", "kmer")
  nanopore_data$cRmet <- nanopore_data$Rmet
} else if (substr(name_sample2, 1, 9)=='megalodon'){
  colnames(nanopore_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3", 
                                "na4", "na5", "coverage", "rpercent")
  nanopore_data$Rmet <- nanopore_data$rpercent / 100
} else if (substr(name_sample2, 1, 5)=='tombo'){
  colnames(nanopore_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3",
                                "na4", "na5", "coverage", "rpercent")
  nanopore_data$Rmet <- nanopore_data$rpercent / 100
} else if(substr(name_sample2, 1, 13)=='bisulfite_bed'){
  colnames(nanopore_data) <- c("chromosome", "pos", "pend", "na1", "na2", "strand", "na3",
                                "na4", "na5", "coverage", "rpercent")
  nanopore_data$Rmet <- nanopore_data$rpercent / 100
  nanopore_data$cRmet <- nanopore_data$Rmet
} else if(substr(name_sample2, 1, 7)=='methccs'){
  colnames(nanopore_data) <- c("chromosome", "pos", "strand",
                                "prob0", "prob1", "met", "unmet", "coverage",
                                "Rmet", "kmer")
  nanopore_data$cRmet <- nanopore_data$Rmet
}

chromid = "all_contigs"
if(allsites=="yes"){
  chromid = "all_contigs"
}else{
  chromid = "main_contigs"
  # chroms_all = unique(bisulfite_data$chromosome)
  chroms <- c("chr1", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
              "chr18", "chr19", "chr2", "chr20", "chr21", "chr22", "chr3", "chr4",  "chr5",
              "chr6", "chr7", "chr8", "chr9", "chrX", "chrY")
  bisulfite_data <- bisulfite_data[bisulfite_data$chromosome %in% chroms, ]
  nanopore_data <- nanopore_data[nanopore_data$chromosome %in% chroms, ]
}

fileprefix = paste(resultdir, '/',
                   paste(datacov, name_sample2, 'vs', name_sample1, midfix, chromid, sep = '_'), 
                   sep='')

bisulfite_data$key <- paste(bisulfite_data$chromosome, bisulfite_data$pos, sep=" ")
nanopore_data$key <- paste(nanopore_data$chromosome, nanopore_data$pos, sep=" ")


stats_result <- list()
# stats ==================================
# site intersect ====
message(sprintf("total CGs in %s: %d", chromid, 0))
message(sprintf("total CGs in %s, %s: %d", chromid, name_sample1, nrow(bisulfite_data)))
message(sprintf("total CGs in %s, %s: %d", chromid, name_sample2, nrow(nanopore_data)))
stats_result['total_cgs_bs'] <- nrow(bisulfite_data)
stats_result['total_cgs_nano'] <- nrow(nanopore_data)

# intersected
pos_inter <- intersect(bisulfite_data$key, nanopore_data$key)
pos_diff_bi <- setdiff(bisulfite_data$key, nanopore_data$key)
pos_diff_na <- setdiff(nanopore_data$key, bisulfite_data$key)
message(sprintf("total CGs in %s, in both: %d", chromid, length(pos_inter)))
message(sprintf("CGs only in %s, in %s: %d", chromid, name_sample1, length(pos_diff_bi)))
message(sprintf("CGs only in %s, in %s: %d", chromid, name_sample2, length(pos_diff_na)))
stats_result['total_cgs_inter'] <- length(pos_inter)
stats_result['total_cgs_diffbs'] <- length(pos_diff_bi)
stats_result['total_cgs_diffnano'] <- length(pos_diff_na)

cov_cf = 100
message(sprintf("coverage stats======="))
message(sprintf("%s CGs(coverage>%d) in %s: %d", chromid, cov_cf, name_sample1, nrow(bisulfite_data[bisulfite_data$coverage > cov_cf,])))
message(sprintf("%s CGs(coverage>%d) in %s: %d", chromid, cov_cf, name_sample2, nrow(nanopore_data[nanopore_data$coverage > cov_cf,])))
message(sprintf("%s CGs mean coverage in %s: %f", chromid, name_sample1, mean(bisulfite_data$coverage)))
message(sprintf("%s CGs mean coverage in %s: %f", chromid, name_sample2, mean(nanopore_data$coverage)))
coverage_bis <- bisulfite_data[bisulfite_data$coverage<=cov_cf,]$coverage
coverage_nan <- nanopore_data[nanopore_data$coverage<=cov_cf,]$coverage
message(sprintf("%s CGs (coverage<=%d) mean coverage in %s: %f", chromid, cov_cf, name_sample1, mean(coverage_bis)))
message(sprintf("%s CGs (coverage<=%d) mean coverage in %s: %f", chromid, cov_cf, name_sample2, mean(coverage_nan)))
# coverage ===========
if(is_plot=='yes'){
  coverage_bind <-data.frame(coverage = c(coverage_bis, coverage_nan), 
                             data_source = c(rep(name_sample1, length(coverage_bis)), 
                                             rep(name_sample2, length(coverage_nan))))
  pdf(paste(fileprefix, 'coverage.pdf', sep = '_'), 10, 6)
  p_cover <- ggplot(coverage_bind, 
                    aes(x=coverage, fill=data_source)) + 
    geom_histogram(binwidth = 1, position = "dodge") + 
    theme_bw() + 
    scale_fill_discrete(breaks=c(name_sample1, name_sample2), 
                        labels=c(paste('', name_sample1, ' ', sep = ' '), 
                                 paste('', name_sample2, sep = ' '))) +
    theme(legend.position = "bottom", 
          legend.title = element_blank(), 
          legend.text = element_text(size=17), 
          axis.title = element_text(size=17), 
          axis.text.x = element_text(size=17), 
          axis.text.y = element_text(size=17), 
          plot.title = element_text(size=18)) +
    ggtitle(paste(datacov, chromid, "coverage", sep = ' '))
  # message(sprintf("plotting======="))
  print(p_cover)
  dev.off()
}

# Rmet ===========================
# rmet global ====
message(sprintf("rmet global stats======="))
covcf = 5
bisulfite_sel <- bisulfite_data[bisulfite_data$coverage>=covcf, 
                                c("key", "cRmet")]
nanopore_sel <- nanopore_data[nanopore_data$coverage>=covcf, 
                              c("key", "Rmet")]
cgkeys_cov5_inter <- intersect(bisulfite_sel$key, nanopore_sel$key)
# ==
cgcov5_bs_diff_len <- length(setdiff(bisulfite_sel$key, nanopore_sel$key))
cgcov5_nano_diff_len <- length(setdiff(nanopore_sel$key, bisulfite_sel$key))
message(sprintf("total CGs coverage >= %d in %s: %d", covcf, name_sample1, 
                nrow(bisulfite_sel)))
message(sprintf("total CGs coverage >= %d in %s: %d", covcf, name_sample2, 
                nrow(nanopore_sel)))
message(sprintf("coverage >= %d, only in %s: %d, only in %s: %d, intersected: %d", 
                covcf, name_sample1, cgcov5_bs_diff_len, name_sample2, 
                cgcov5_nano_diff_len, 
                length(cgkeys_cov5_inter)))
stats_result['cgs_cov5_bs'] <- nrow(bisulfite_sel)
stats_result['cgs_cov5_nano'] <- nrow(nanopore_sel)
stats_result['cgs_cov5_inter'] <- length(cgkeys_cov5_inter)
stats_result['cgs_cov5_diffbs'] <- cgcov5_bs_diff_len
stats_result['cgs_cov5_diffnano'] <- cgcov5_nano_diff_len

bisulfite_sel0 <- bisulfite_sel[bisulfite_sel$cRmet<=0, ]$key
bisulfite_sel01 <- bisulfite_sel[bisulfite_sel$cRmet<=0.1, ]$key
nanopore_sel0 <- nanopore_sel[nanopore_sel$Rmet==0, ]$key
nanopore_sel01 <- nanopore_sel[nanopore_sel$Rmet<=0.1, ]$key

cgcov5_0_inter_len <- length(intersect(bisulfite_sel0, nanopore_sel0))
cgcov5_0_diffbs_len <- length(setdiff(bisulfite_sel0, nanopore_sel0))
cgcov5_0_diffnano_len <- length(setdiff(nanopore_sel0, bisulfite_sel0))
message(sprintf("total rmet 0 CGs coverage >= %d in %s: %d", covcf, name_sample1, 
                length(bisulfite_sel0)))
message(sprintf("total rmet 0 CGs coverage >= %d in %s: %d", covcf, name_sample2, 
                length(nanopore_sel0)))
message(sprintf("rmet 0 coverage >= %d, only in %s: %d, only in %s: %d, intersected: %d", 
                covcf, name_sample1, cgcov5_0_diffbs_len, name_sample2,
                cgcov5_0_diffnano_len, 
                cgcov5_0_inter_len))
stats_result['cgs0_cov5_bs'] <- length(bisulfite_sel0)
stats_result['cgs0_cov5_nano'] <- length(nanopore_sel0)
stats_result['cgs0_cov5_inter'] <- cgcov5_0_inter_len
stats_result['cgs0_cov5_diffbs'] <- cgcov5_0_diffbs_len
stats_result['cgs0_cov5_diffnano'] <- cgcov5_0_diffnano_len

cgcov5_0_01_inter_len <- length(intersect(bisulfite_sel0, nanopore_sel01))
cgcov5_0_01_diffbs_len <- length(setdiff(bisulfite_sel0, nanopore_sel01))
cgcov5_0_01_diffnano_len <- length(setdiff(nanopore_sel01, bisulfite_sel0))
message(sprintf("total rmet 0 CGs coverage >= %d in %s: %d", covcf, name_sample1, 
                length(bisulfite_sel0)))
message(sprintf("total rmet <=0.1 CGs coverage >= %d in %s: %d", covcf, name_sample2, 
                length(nanopore_sel01)))
message(sprintf("rmet 0/<=0.1 coverage >= %d, only in %s: %d, only in %s: %d, intersected: %d", 
                covcf, name_sample1, cgcov5_0_01_diffbs_len, name_sample2,
                cgcov5_0_01_diffnano_len, 
                cgcov5_0_01_inter_len))
stats_result['cgs0_01_cov5_bs'] <- length(bisulfite_sel0)
stats_result['cgs0_01_cov5_nano'] <- length(nanopore_sel01)
stats_result['cgs0_01_cov5_inter'] <- cgcov5_0_01_inter_len
stats_result['cgs0_01_cov5_diffbs'] <- cgcov5_0_01_diffbs_len
stats_result['cgs0_01_cov5_diffnano'] <- cgcov5_0_01_diffnano_len

cgcov5_01_01_inter_len <- length(intersect(bisulfite_sel01, nanopore_sel01))
cgcov5_01_01_diffbs_len <- length(setdiff(bisulfite_sel01, nanopore_sel01))
cgcov5_01_01_diffnano_len <- length(setdiff(nanopore_sel01, bisulfite_sel01))
message(sprintf("total rmet <=0.1 CGs coverage >= %d in %s: %d", covcf, name_sample1, 
                length(bisulfite_sel01)))
message(sprintf("total rmet <=0.1 CGs coverage >= %d in %s: %d", covcf, name_sample2, 
                length(nanopore_sel01)))
message(sprintf("rmet <=0.1 coverage >= %d, only in %s: %d, only in %s: %d, intersected: %d", 
                covcf, name_sample1, cgcov5_01_01_diffbs_len, name_sample2,
                cgcov5_01_01_diffnano_len, 
                cgcov5_01_01_inter_len))
stats_result['cgs01_01_cov5_bs'] <- length(bisulfite_sel01)
stats_result['cgs01_01_cov5_nano'] <- length(nanopore_sel01)
stats_result['cgs01_01_cov5_inter'] <- cgcov5_01_01_inter_len
stats_result['cgs01_01_cov5_diffbs'] <- cgcov5_01_01_diffbs_len
stats_result['cgs01_01_cov5_diffnano'] <- cgcov5_01_01_diffnano_len


bisulfite_sel1 <- bisulfite_sel[bisulfite_sel$cRmet==1, ]$key
bisulfite_sel09 <- bisulfite_sel[bisulfite_sel$cRmet>=0.9, ]$key
nanopore_sel1 <- nanopore_sel[nanopore_sel$Rmet==1, ]$key
nanopore_sel09 <- nanopore_sel[nanopore_sel$Rmet>=0.9, ]$key

cgcov5_1_inter_len <- length(intersect(bisulfite_sel1, nanopore_sel1))
cgcov5_1_diffbs_len <- length(setdiff(bisulfite_sel1, nanopore_sel1))
cgcov5_1_diffnano_len <- length(setdiff(nanopore_sel1, bisulfite_sel1))
message(sprintf("total rmet 1 CGs coverage >= %d in %s: %d", covcf, name_sample1, 
                length(bisulfite_sel1)))
message(sprintf("total rmet 1 CGs coverage >= %d in %s: %d", covcf, name_sample2, 
                length(nanopore_sel1)))
message(sprintf("rmet 1 coverage >= %d, only in %s: %d, only in %s: %d, intersected: %d", 
                covcf, name_sample1, cgcov5_1_diffbs_len, name_sample2, 
                cgcov5_1_diffnano_len, 
                cgcov5_1_inter_len))
stats_result['cgs1_cov5_bs'] <- length(bisulfite_sel1)
stats_result['cgs1_cov5_nano'] <- length(nanopore_sel1)
stats_result['cgs1_cov5_inter'] <- cgcov5_1_inter_len
stats_result['cgs1_cov5_diffbs'] <- cgcov5_1_diffbs_len
stats_result['cgs1_cov5_diffnano'] <- cgcov5_1_diffnano_len

cgcov5_1_09_inter_len <- length(intersect(bisulfite_sel1, nanopore_sel09))
cgcov5_1_09_diffbs_len <- length(setdiff(bisulfite_sel1, nanopore_sel09))
cgcov5_1_09_diffnano_len <- length(setdiff(nanopore_sel09, bisulfite_sel1))
message(sprintf("total rmet 1 CGs coverage >= %d in %s: %d", covcf, name_sample1, 
                length(bisulfite_sel1)))
message(sprintf("total rmet >=0.9 CGs coverage >= %d in %s: %d", covcf, name_sample2, 
                length(nanopore_sel09)))
message(sprintf("rmet 1/>=0.9 coverage >= %d, only in %s: %d, only in %s: %d, intersected: %d", 
                covcf, name_sample1, cgcov5_1_09_diffbs_len, name_sample2, 
                cgcov5_1_09_diffnano_len, 
                cgcov5_1_09_inter_len))
stats_result['cgs1_09_cov5_bs'] <- length(bisulfite_sel1)
stats_result['cgs1_09_cov5_nano'] <- length(nanopore_sel09)
stats_result['cgs1_09_cov5_inter'] <- cgcov5_1_09_inter_len
stats_result['cgs1_09_cov5_diffbs'] <- cgcov5_1_09_diffbs_len
stats_result['cgs1_09_cov5_diffnano'] <- cgcov5_1_09_diffnano_len

cgcov5_09_09_inter_len <- length(intersect(bisulfite_sel09, nanopore_sel09))
cgcov5_09_09_diffbs_len <- length(setdiff(bisulfite_sel09, nanopore_sel09))
cgcov5_09_09_diffnano_len <- length(setdiff(nanopore_sel09, bisulfite_sel09))
message(sprintf("total rmet >=0.9 CGs coverage >= %d in %s: %d", covcf, name_sample1, 
                length(bisulfite_sel09)))
message(sprintf("total rmet >=0.9 CGs coverage >= %d in %s: %d", covcf, name_sample2, 
                length(nanopore_sel09)))
message(sprintf("rmet >=0.9 coverage >= %d, only in %s: %d, only in %s: %d, intersected: %d", 
                covcf, name_sample1, cgcov5_09_09_diffbs_len, name_sample2, 
                cgcov5_09_09_diffnano_len, 
                cgcov5_09_09_inter_len))
stats_result['cgs09_09_cov5_bs'] <- length(bisulfite_sel09)
stats_result['cgs09_09_cov5_nano'] <- length(nanopore_sel09)
stats_result['cgs09_09_cov5_inter'] <- cgcov5_09_09_inter_len
stats_result['cgs09_09_cov5_diffbs'] <- cgcov5_09_09_diffbs_len
stats_result['cgs09_09_cov5_diffnano'] <- cgcov5_09_09_diffnano_len


if(is_plot=='yes'){
  pdf(paste(fileprefix, 'rmet_global.pdf', sep = '_'), 9, 6)
  widths = c(0.01, 0.05, 0.1)
  for(width in widths){
    # plot ==
    rmet_bis <- bisulfite_sel$cRmet
    rmet_bis[rmet_bis<0] = 0
    rmet_bis <- ratio_stats(rmet_bis, 0, 1, width)
    
    rmet_nan <- nanopore_sel$Rmet
    rmet_nan <- ratio_stats(rmet_nan, 0, 1, width)
    
    message(sprintf("Rmet, cov>=%d, %s vs %s: pearson corr: %f, spearman corr: %f", 
                    covcf, name_sample1, name_sample2, 
                    cor(rmet_bis$ratio, rmet_nan$ratio, method='pearson'), 
                    cor(rmet_bis$ratio, rmet_nan$ratio, method='spearman')))
    
    rmet_bis$ssample <- rep(name_sample1, nrow(rmet_bis))
    rmet_nan$ssample <- rep(name_sample2, nrow(rmet_nan))
    rmet_bind <- rbind.data.frame(rmet_bis, rmet_nan)
    idxmin <- min(rmet_bis$idx)
    idxmax <- max(rmet_bis$idx)
    # print(idxmin)
    # print(idxmax)
    p_rmet <- ggplot(rmet_bind, aes(x=idx, y=ratio, fill=ssample)) + 
      geom_bar(stat="identity", position=position_dodge()) + 
      theme_bw() + 
      scale_fill_discrete(breaks=c(name_sample1, name_sample2), 
                          labels=c(paste('', name_sample1, ' ', sep = ' '), 
                                   paste('', name_sample2, sep = ' '))) +
      theme(legend.position = "bottom", 
            legend.title = element_blank(), 
            legend.text = element_text(size=16), 
            axis.title = element_text(size=16), 
            axis.text.x = element_text(size=16), 
            axis.text.y = element_text(size=16), 
            plot.title = element_text(size=17)) +
      ggtitle(paste(datacov, chromid, " (coverage >=", covcf, ", pearson correlation:", 
                    round(cor(rmet_bis$ratio, rmet_nan$ratio, method='pearson'), 4), 
                    ")", sep = ' ')) + 
      scale_x_continuous(name="Rmet", 
                         breaks=seq(0, 1/width, 10 /(width/0.01)), 
                         labels=seq(0, 1, 1/10))+ 
      scale_y_continuous(name='percentage')
    scale_fill_brewer(palette="Set1")
    print(p_rmet)
  }
  dev.off()
}


# Rmet point level ===
message(sprintf("rmet point level stats======="))
bisulfite_sel_rp <- bisulfite_sel[bisulfite_sel$key %in% cgkeys_cov5_inter, ]
bisulfite_sel_rp <- bisulfite_sel_rp[order(bisulfite_sel_rp$key), ]
nanopore_sel_rp <- nanopore_sel[nanopore_sel$key %in% cgkeys_cov5_inter, ]
nanopore_sel_rp <- nanopore_sel_rp[order(nanopore_sel_rp$key), ]

rmet_bis <- bisulfite_sel_rp$cRmet
rmet_bis[rmet_bis<0] = 0
rmet_nan <- nanopore_sel_rp$Rmet

stats_result['rmet_pearson'] <- cor(rmet_bis, rmet_nan, method='pearson') 
stats_result['rmet_spearman'] <- cor(rmet_bis, rmet_nan, method='spearman')

if(is_plot == 'yes'){
  pdf(paste(fileprefix, 'rmet_x.pdf', sep = '_'), 9, 6)
  # scale 10
  for(nunit in c(10, 25, 40)){
    rmet_mat_10 <- as.data.frame(generate_rtype_matrix(rmet_nan, rmet_bis, nunit))
    rmet_mat_10[rmet_mat_10==0] = 1
    sim.df <- cbind(ID=as.numeric(rownames(rmet_mat_10)), rmet_mat_10)
    sim.df <- melt(sim.df, id.vars = 'ID')
    sim.df$variable <- as.numeric(as.character(sim.df$variable))
    sim.df$label <- as.character(round(sim.df$value, 2))
    
    maxvalue = max(sim.df$value)
    if(maxvalue < 1000000){
      maxvalue = 1000000
    }
    
    if(nunit==10){
      print(heatplot(rmet_mat_10,xlab = name_sample1, ylab = name_sample2))
    }else{
      p <- ggplot(sim.df, aes(variable, ID, fill=value)) + 
        geom_raster(hjust = 0, vjust = 0) +
        theme_bw() + 
        geom_hline(yintercept = 0) + 
        geom_vline(xintercept = 0) +
        scale_fill_gradient2(low = "#006d2c", high = "#bd0026", mid = "#feb24c",
                             midpoint = log10(maxvalue)/2, 
                             limit = c(1, maxvalue), 
                             space = "Lab", name="count", trans="log10", 
                             labels=comma) + 
        scale_x_continuous(breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
        scale_y_continuous(breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
        theme(axis.title = element_text(size = 16), 
              axis.text=element_text(size=16), 
              legend.text = element_text(size=16), 
              legend.title = element_text(size=16)) +
        xlab(name_sample1) + ylab(name_sample2) +
        ggtitle(paste(rmetfile2label, "(pearson correlation:", 
                      paste(round(cor(rmet_bis, rmet_nan, method='pearson'), 4), 
                            ")", sep = ""), 
                      sep = ' '))
      print(p)
    }
  }
  # print(heatplot(rmet_mat_10,xlab = name_sample1, ylab = name_sample2))
  dev.off()
}

# correlation calculation ==
a = 1
while(a){
  disaplay_corr_eq(rmet_bis, rmet_nan, 0, 1)
  disaplay_corr_neq(rmet_bis, rmet_nan, 0, 1)
  # disaplay_corr_eq(rmet_bis, rmet_nan, 0, 0)
  disaplay_corr_eq(rmet_bis, rmet_nan, 0, 0.3)
  disaplay_corr_ueq(rmet_bis, rmet_nan, 0, 0.3)
  disaplay_corr_ueq(rmet_bis, rmet_nan, 0.3, 0.7)
  disaplay_corr_ueq(rmet_bis, rmet_nan, 0.7, 1)
  disaplay_corr_neq(rmet_bis, rmet_nan, 0.7, 1)
  # disaplay_corr_eq(rmet_bis, rmet_nan, 1, 1)
  a = 0
}

result_names <- names(stats_result)
stats_result <- as.data.frame(stats_result)
colnames(stats_result) <- result_names
write.table(stats_result, paste(resultdir, '/', resultname, sep = ''), quote = F, 
            row.names = F, sep = '\t')







