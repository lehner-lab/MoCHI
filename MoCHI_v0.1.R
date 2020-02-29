#!/usr/bin/env Rscript

### Compute all complete sublandscapes
### Obtain all epistatic terms, background relative and background averaged terms

###########################
### COMMAND-LINE OPTIONS
###########################

option_list <- list(
  optparse::make_option(opt_str=c("--workspacePath", "-w"), help = "Path to DiMSum workspace .RData file after successful completion of stage 7"),
  optparse::make_option(opt_str=c("--maxOrder"), type="integer", default=2, help = "Maximum number of nucleotide or amino acid substitutions for coding or non-coding sequences respectively (default:2)"),
  optparse::make_option(opt_str=c("--startStage", "-s"), type="integer", default=1, help = "Start at a specified pipeline stage"),
  optparse::make_option(opt_str=c("--stopStage", "-t"), type="integer", default=0, help = "Stop at a specified pipeline stage (default: 0 i.e. no stop condition)"),
  optparse::make_option(opt_str=c("--numCores", "-c"), type="integer", default=1, help = "Number of available CPU cores")  
)

arg_list <- optparse::parse_args(optparse::OptionParser(option_list=option_list))
#Display arguments
message(paste("\n\n\n*******", "MoCHI wrapper command-line arguments", "*******\n\n\n"))
message(paste(formatDL(unlist(arg_list)), collapse = "\n"))

###########################
### LOAD DIMSUM PACKAGE
###########################

library(MoCHI)

###########################
### RUN
###########################

#Load data
load(arg_list[["workspacePath"]])
dimsum_meta <- pipeline[['7_fitness']]
dimsum_meta[['numCores']] <- arg_list[["numCores"]]
dimsum_meta[['HOEStartStage']] <- arg_list[["startStage"]]
dimsum_meta[['HOEStopStage']] <- arg_list[["stopStage"]]

#Determine combinatorial complete landscapes
mochi_stage_combinatorial_complete_landscapes(
  dimsum_meta = dimsum_meta,
  epistasis_outpath = file.path(dimsum_meta[["tmp_path"]], "epistasis"),
  order_max = arg_list[["maxOrder"]],
  report_outpath = file.path(dimsum_meta[["tmp_path"]], "epistasis"))

#Obtain all epistatic terms, background relative and background averaged terms
mochi_stage_higher_order_epistasis(
  dimsum_meta = dimsum_meta,
  epistasis_outpath = file.path(dimsum_meta[["tmp_path"]], "epistasis"),
  report_outpath = file.path(dimsum_meta[["tmp_path"]], "epistasis"))

