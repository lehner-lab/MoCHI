#!/usr/bin/env Rscript

###########################
### COMMAND-LINE OPTIONS
###########################

option_list <- list(
  optparse::make_option(opt_str=c("--workspacePath", "-w"), help = "Path to DiMSum workspace .RData file after successful completion of stage 7"),
  optparse::make_option(opt_str=c("--inputFile", "-i"), help = "Path to plain text file with 'WT', 'fitness', 'sigma' and 'nt_seq' or ('aa_seq' and 'STOP') columns"),
  optparse::make_option(opt_str=c("--outputPath", "-o"), help = "Path to directory to use for output files (required if inputFile supplied)"),
  optparse::make_option(opt_str=c("--projectName", "-p"), help = "Project name (required if inputFile supplied)"),
  optparse::make_option(opt_str=c("--analysisName", "-a"), help = "Analysis (or run) name (default:'mochi')"),
  optparse::make_option(opt_str=c("--startStage", "-s"), type="integer", default=0, help = "Start at a specified pipeline stage (default:0)"),
  optparse::make_option(opt_str=c("--stopStage", "-t"), type="integer", default=2, help = "Stop at a specified pipeline stage (default:2)"),
  optparse::make_option(opt_str=c("--numCores", "-c"), type="integer", default=1, help = "Number of available CPU cores"),
  optparse::make_option(opt_str=c("--maxOrder"), type="integer", default=2, help = "Maximum number of nucleotide or amino acid substitutions for coding or non-coding sequences respectively (default:2)"),
  optparse::make_option(opt_str=c("--numReplicates"), type="integer", default=2, help = "Number of biological replicates (or sample size) from which fitness and error estimates derived. Used to calculate degrees of freedom if 'testType' is 'ttest' (default:2)"),
  optparse::make_option(opt_str=c("--significanceThreshold"), type="double", default=0.1, help = "Significance threshold for specific and background averaged terms (default:0.1; 1 disables the treshold)"),
  optparse::make_option(opt_str=c("--adjustmentMethod"), default="fdr", help = "Method for adjusting P-values for multiple comparisons; any method in p.adjust.methods is permitted (default:'fdr')"),
  optparse::make_option(opt_str=c("--testType"), default="ztest", help = "Type of statistical test to use: either 'ztest' or 'ttest' (default:'ztest')"),
  optparse::make_option(opt_str=c("--orderSubset"), default="all", help = "Comma-separated list of (integer) orders of epistatic terms to retain for fitness reconstruction or 'all' (default:'all')"),
  optparse::make_option(opt_str=c("--inferAdditiveTrait"), default=F, help = "Infer additive trait and remove associated global epistasis before calculating higher order genetic interactions (default:F)")
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
  workspacePath=arg_list[["workspacePath"]],
  inputFile=arg_list[["inputFile"]],
  outputPath=arg_list[["outputPath"]],
  projectName=arg_list[["projectName"]],
  analysisName=arg_list[["analysisName"]],
  startStage=arg_list[["startStage"]],
  stopStage=arg_list[["stopStage"]],
  numCores=arg_list[["numCores"]],
  maxOrder=arg_list[["maxOrder"]],
  numReplicates=arg_list[["numReplicates"]],
  significanceThreshold=arg_list[["significanceThreshold"]],
  adjustmentMethod=arg_list[["adjustmentMethod"]],
  testType=arg_list[["testType"]],
  orderSubset=arg_list[["orderSubset"]],
  inferAdditiveTrait=arg_list[["inferAdditiveTrait"]]
  )
