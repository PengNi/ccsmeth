#! /usr/bin/env Rscript
# coding=utf-8

# Copyright (C) 2020  Vahid Akbari

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

library(sys)
suppressPackageStartupMessages(library("R.utils"))
if (!require('DSS')) install.packages('DSS')
suppressPackageStartupMessages(library("DSS"))

#The first argument is first file, second is second file and third is your
#output name, forth argument is the  dis.merge in callDMR function, fifth
#argument is the minlen in callDMR function and the sixth argument is the minCG
#in callDMR function. the DMA will be first file vs second file.  So your first
#file should be case and second shoud be control
args <- commandArgs(trailingOnly = TRUE)
print("######################################################################################################################")
print("DSS Arguments:")
print("Case:")
print(args[1])
print("Control:")
print(args[2])
print("Output:")
print(args[3])
print("dis_merge:")
print(args[4])
print("min_len")
print(args[5])
print("min_CG")
print(args[6])
print("Smoothing_span")
print(args[7])
print("smoothing_flag")
print(args[8])
print("pval_cutoff")
print(args[9])
print("delta")
print(args[10])
print("pct_sig")
print(args[11])
print("equal_disp")
print(args[12])
print("######################################################################################################################")
print("reading files")

list_case= as.list(unlist(strsplit(args[1], ',')))
list_control= as.list(unlist(strsplit(args[2], ',')))
input_list= list()
input_vector_list= list()
input_list_case= list()
input_list_control= list()
index_list=0
for (n in 1:length(list_case)) {
  index_list= index_list+1
  case_num<- paste("C",as.character(n),sep = "")
  input_vector_list[[index_list]]<- case_num
  input_list_case[[n]]<- case_num
}

for (n in 1:length(list_control)) {
  index_list= index_list+1
  control_num<- paste("N",as.character(n),sep = "")
  input_vector_list[[index_list]]<- control_num
  input_list_control[[n]]<- control_num
}

All_files= paste(args[1],args[2],sep = ",")
All_files= as.list(unlist(strsplit(All_files, ',')))
for (n in 1:length(All_files)) {
  file= as.character(All_files[n])
  input_list[[n]] <- read.table(file,header=TRUE,sep = '\t',col.names = c("chr","pos","N","X"))
}

dis_merge=as.integer(args[4])
min_len=as.integer(args[5])
min_CG=as.integer(args[6])
output=args[3]
ss=as.integer(args[7])
sf=args[8]

pv=as.double(args[9])
del=as.double(args[10])
pct=as.double(args[11])
ed=args[12]
if (ed == "FALSE"){ ed= FALSE }
if (ed == "TRUE"){ ed= TRUE }

DMCpG_results=paste(output,"_DMLtest.txt",sep = "")
DMLocus_results=paste(output,"_callDML.txt",sep = "")
DMRegion_results=paste(output,"_callDMR.txt",sep = "")

#DM analysis on BS and Nanopore methylation By DSS
# manual http://bioconductor.org/packages/release/bioc/manuals/DSS/man/DSS.pdf
print("DMA process started")
DSObject<- makeBSseqData(input_list,as.vector(unlist(input_vector_list, use.names=FALSE)))

if (sf == "FALSE"){
  print("DMLtest, smoothing=FALSE")
  test<- DMLtest(DSObject, group1=as.vector(unlist(input_list_case, use.names=FALSE)),
                 group2=as.vector(unlist(input_list_control, use.names=FALSE)), equal.disp = ed,
                 smoothing = FALSE)
  }
if (sf == "TRUE"){
  print("DMLtest, smoothing=TRUE")
  test<- DMLtest(DSObject, group1=as.vector(unlist(input_list_case, use.names=FALSE)),
               group2=as.vector(unlist(input_list_control, use.names=FALSE)), equal.disp = ed,
               smoothing = TRUE, smoothing.span = ss)
  }
write.table(test,DMCpG_results, sep="\t", row.names=F, quote=F)

DM_loci<- callDML(test, delta=del, p.threshold=pv)
write.table(DM_loci, DMLocus_results, sep="\t", row.names=F, quote=F)

DM_region<- callDMR(test, delta=del, p.threshold=pv, minlen=min_len, minCG=min_CG, dis.merge=dis_merge, pct.sig=pct)
write.table(DM_region,DMRegion_results, sep="\t", row.names=F, quote=F)
