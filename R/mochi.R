#' Run MoCHI pipeline
#'
#' This function runs the MoCHI pipeline.
#'
#' @param workspacePath Path to DiMSum workspace .RData file after successful completion of stage 7
#' @param inputFile Path to plain text file with 'WT', 'fitness', 'sigma' and 'nt_seq' or ('aa_seq' and 'STOP') columns
#' @param outputPath Path to directory to use for output files
#' @param projectName Project name
#' @param startStage Start at a specified pipeline stage (default:1)
#' @param stopStage Stop at a specified pipeline stage (default:0 i.e. no stop condition)
#' @param numCores Number of available CPU cores (default:1)
#' @param maxOrder Maximum number of nucleotide or amino acid substitutions for coding or non-coding sequences respectively (default:2)
#' @param numReplicates Number of biological replicates (or sample size) from which fitness and error estimates derived. Used to calculate degrees of freedom if 'testType' is 'ttest' (default:2)
#' @param FDR_threshold FDR threshold for significant specific and background averaged terms (default:0.1; 1 disables the treshold)
#' @param testType Type of statistical test to use: either 'ztest' or 'ttest' (default:'ztest')
#'
#' @return Nothing
#' @export
#' @import data.table
mochi <- function(
  workspacePath=NULL,
  inputFile=NULL,
  outputPath=NULL,
  projectName=NULL,
  startStage=1,
  stopStage=0,
  numCores=1,
  maxOrder=2,
  numReplicates=2,
  FDR_threshold=0.1,
  testType="ztest"
  ){

  #Display welcome
  message(paste("\n\n\n*******", "Running MoCHI pipeline", "*******\n\n\n"))
  message(paste(formatDL(unlist(list("Package version" = as.character(packageVersion("MoCHI"))))), collapse = "\n"))
  message(paste(formatDL(unlist(list("R version" = version$version.string))), collapse = "\n"))

  ### Basic checks
  ###########################


  ### Setup
  ###########################

  mochi_arg_list <- list(
    "workspacePath" = list(workspacePath, c("character", "NULL")), #file exists (if not NULL) -- checked in dimsum__validate_input
    "inputFile" = list(inputFile, c("character", "NULL")), #file exists (if not NULL) -- checked in dimsum__validate_input
    "outputPath" = list(outputPath, c("character", "NULL")), #directory exists (if not NULL) -- checked in mochi__validate_input
    "projectName" = list(projectName, c("character", "NULL")), #character string (if not NULL) -- checked in mochi__validate_input
    "mochiStartStage" = list(startStage, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "mochiStopStage" = list(stopStage, c("integer")), #positive integer (zero inclusive) -- checked in mochi__validate_input
    "numCores" = list(numCores, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "maxOrder" = list(maxOrder, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "numReplicates" = list(numReplicates, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "FDR_threshold" = list(FDR_threshold, c("double")), #positive double less than 1 (zero exclusive) -- checked in mochi__validate_input
    "testType" = list(testType, c("character")) #character string (either 'ttest' or 'ztest') -- checked in mochi__validate_input
    )

  #Validate input
  exp_metadata <- mochi__validate_input(mochi_arg_list)

  #If input file supplied
  if(!is.null(exp_metadata[["inputFile"]])){
    #Create project directory (if doesn't already exist)
    exp_metadata[["project_path"]] <- file.path(exp_metadata[["outputPath"]], exp_metadata[["projectName"]])
    exp_metadata[["fitness_path"]] <- exp_metadata[["project_path"]]
    suppressWarnings(dir.create(exp_metadata[["project_path"]]))
    #Create temp directory (if doesn't already exist)
    exp_metadata[["tmp_path"]] <- file.path(exp_metadata[["project_path"]], "tmp")
    suppressWarnings(dir.create(exp_metadata[["tmp_path"]]))

    #Copy inputFile to project_path as RData file
    if(exp_metadata[["sequenceType"]]=="noncoding"){
      all_variants <- fread(exp_metadata[["inputFile"]], colClasses = c(nt_seq = "character"))
    }else if(exp_metadata[["sequenceType"]]=="coding"){
      all_variants <- fread(exp_metadata[["inputFile"]], colClasses = c(aa_seq = "character"))
    }
    save(all_variants, file = file.path(exp_metadata[["project_path"]], paste0(exp_metadata[["projectName"]], '_fitness_replicates.RData')))
  }

  #If workspace file supplied
  if(!is.null(exp_metadata[["workspacePath"]])){
    #Save variables overwritten by loading workspace
    mochiStartStage <- startStage
    mochiStopStage <- stopStage
    mochiNumCores <- numCores
    #Load data
    load(exp_metadata[["workspacePath"]])
    if(!'5_analyse' %in% names(pipeline)){
      stop(paste0("Invalid '", "workspacePath", "' argument (stage 5)"), call. = FALSE)
    }
    exp_metadata <- pipeline[['5_analyse']]
    exp_metadata[['mochiStartStage']] <- mochiStartStage
    exp_metadata[['mochiStopStage']] <- mochiStopStage
    exp_metadata[['numCores']] <- mochiNumCores
    exp_metadata[['maxOrder']] <- maxOrder
    exp_metadata[['numReplicates']] <- numReplicates
    exp_metadata[['FDR_threshold']] <- FDR_threshold
    exp_metadata[['testType']] <- testType
  }

  ### Pipeline stages
  ###########################

  #Determine combinatorial complete landscapes
  mochi_stage_combinatorial_complete_landscapes(
    dimsum_meta = exp_metadata,
    epistasis_outpath = file.path(exp_metadata[["tmp_path"]], "epistasis"),
    order_max = exp_metadata[["maxOrder"]],
    report_outpath = file.path(exp_metadata[["tmp_path"]], "epistasis"))

  #Obtain all epistatic terms, background relative and background averaged terms
  mochi_stage_higher_order_epistasis(
    dimsum_meta = exp_metadata,
    epistasis_outpath = file.path(exp_metadata[["tmp_path"]], "epistasis"),
    report_outpath = file.path(exp_metadata[["tmp_path"]], "epistasis"))

  ### Save workspace
  ###########################

  message("Done")

}
