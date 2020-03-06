#!/usr/bin/env Rscript

###########################
### COMMAND-LINE OPTIONS
###########################

option_list <- list(
  optparse::make_option(opt_str=c("--workspacePath", "-w"), help = "Path to DiMSum workspace .RData file after successful completion of stage 7"),
  optparse::make_option(opt_str=c("--inputFile", "-i"), help = "Path to plain text file with 'WT', 'fitness', 'sigma' and 'nt_seq' or ('aa_seq' and 'STOP') columns"),
  optparse::make_option(opt_str=c("--outputPath", "-o"), help = "Path to directory to use for output files"),
  optparse::make_option(opt_str=c("--projectName", "-p"), help = "Project name"),
  optparse::make_option(opt_str=c("--startStage", "-s"), type="integer", default=1, help = "Start at a specified pipeline stage"),
  optparse::make_option(opt_str=c("--stopStage", "-t"), type="integer", default=0, help = "Stop at a specified pipeline stage (default: 0 i.e. no stop condition)"),
  optparse::make_option(opt_str=c("--numCores", "-c"), type="integer", default=1, help = "Number of available CPU cores"),
  optparse::make_option(opt_str=c("--maxOrder"), type="integer", default=2, help = "Maximum number of nucleotide or amino acid substitutions for coding or non-coding sequences respectively (default:2)")
)

arg_list <- optparse::parse_args(optparse::OptionParser(option_list=option_list))
#Display arguments
message(paste("\n\n\n*******", "MoCHI wrapper command-line arguments", "*******\n\n\n"))
message(paste(formatDL(unlist(arg_list)), collapse = "\n"))

###########################
### LOAD MoCHI PACKAGE
###########################

library(MoCHI)

###########################
### RUN
###########################

mochi(
  outputPath=arg_list[["outputPath"]],
  projectName=arg_list[["projectName"]],
  startStage=arg_list[["startStage"]],
  stopStage=arg_list[["stopStage"]],
  numCores=arg_list[["numCores"]],
  maxOrder=arg_list[["maxOrder"]])
