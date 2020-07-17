#' Run MoCHI pipeline
#'
#' This function runs the MoCHI pipeline.
#'
#' @param workspacePath Path to DiMSum workspace .RData file after successful completion of stage 7
#' @param inputFile Path to plain text file with 'WT', 'fitness', 'sigma' and 'nt_seq' or ('aa_seq' and 'STOP') columns
#' @param outputPath Path to directory to use for output files (required if inputFile supplied)
#' @param projectName Project name (required if inputFile supplied)
#' @param analysisName Analysis (or run) name (default:'mochi')
#' @param startStage Start at a specified pipeline stage (default:0)
#' @param stopStage Stop at a specified pipeline stage (default:2)
#' @param numCores Number of available CPU cores (default:1)
#' @param maxOrder Maximum number of nucleotide or amino acid substitutions for coding or non-coding sequences respectively (default:2)
#' @param numReplicates Number of biological replicates (or sample size) from which fitness and error estimates derived. Used to calculate degrees of freedom if 'testType' is 'ttest' (default:2)
#' @param significanceThreshold Significance threshold for specific and background averaged terms (default:0.1; 1 disables the treshold)
#' @param adjustmentMethod Method for adjusting P-values for multiple comparisons; any method in p.adjust.methods is permitted (default:'fdr')
#' @param testType Type of statistical test to use: either 'ztest' or 'ttest' (default:'ztest')
#' @param orderSubset Comma-separated list of (integer) orders of epistatic terms to retain for fitness reconstruction or 'all' (default:'all')
#' @param inferAdditiveTrait Infer additive trait and remove associated global epistasis before calculating higher order genetic interactions (default:F)
#'
#' @return Nothing
#' @export
#' @import data.table
mochi <- function(
  workspacePath=NULL,
  inputFile=NULL,
  outputPath=NULL,
  projectName=NULL,
  analysisName="mochi",
  startStage=0,
  stopStage=2,
  numCores=1,
  maxOrder=2,
  numReplicates=2,
  significanceThreshold=0.1,
  adjustmentMethod="fdr",
  testType="ztest",
  orderSubset="all",
  inferAdditiveTrait=F
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
    "workspacePath" = list(workspacePath, c("character", "NULL")), #file exists (if not NULL) -- checked in mochi__validate_input
    "inputFile" = list(inputFile, c("character", "NULL")), #file exists (if not NULL) -- checked in mochi__validate_input
    "outputPath" = list(outputPath, c("character", "NULL")), #directory exists (if not NULL) -- checked in mochi__validate_input
    "projectName" = list(projectName, c("character", "NULL")), #character string (if not NULL) -- checked in mochi__validate_input
    "analysisName" = list(analysisName, c("character")), #character string -- checked in mochi__validate_input
    "mochiStartStage" = list(startStage, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "mochiStopStage" = list(stopStage, c("integer")), #positive integer (zero inclusive) -- checked in mochi__validate_input
    "numCores" = list(numCores, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "maxOrder" = list(maxOrder, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "numReplicates" = list(numReplicates, c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "significanceThreshold" = list(significanceThreshold, c("double")), #positive double less than 1 (zero exclusive) -- checked in mochi__validate_input
    "adjustmentMethod" = list(adjustmentMethod, c("character")), #character string in p.adjust.methods -- checked in mochi__validate_input
    "testType" = list(testType, c("character")), #character string (either 'ttest' or 'ztest') -- checked in mochi__validate_input
    "orderSubset" = list(orderSubset, c("character")), #comma-separated list of integers or "all" -- checked in mochi__validate_input
    "inferAdditiveTrait" = list(inferAdditiveTrait, c("logical")) #logical -- checked in mochi__validate_input
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
    #Save MoCHI exp_metadata
    exp_metadata_mochi <- exp_metadata
    #Load data
    load(exp_metadata[["workspacePath"]])
    if(!'5_analyse' %in% names(pipeline)){
      stop(paste0("Invalid '", "workspacePath", "' argument (stage 5)"), call. = FALSE)
    }
    #Additional arguments
    exp_metadata <- pipeline[['5_analyse']]
    exp_metadata[['analysisName']] <- exp_metadata_mochi[['analysisName']]
    exp_metadata[['mochiStartStage']] <- exp_metadata_mochi[['mochiStartStage']]
    exp_metadata[['mochiStopStage']] <- exp_metadata_mochi[['mochiStopStage']]
    exp_metadata[['numCores']] <- exp_metadata_mochi[['numCores']]
    exp_metadata[['maxOrder']] <- exp_metadata_mochi[['maxOrder']]
    exp_metadata[['numReplicates']] <- exp_metadata_mochi[['numReplicates']]
    exp_metadata[['significanceThreshold']] <- exp_metadata_mochi[['significanceThreshold']]
    exp_metadata[['adjustmentMethod']] <- exp_metadata_mochi[['adjustmentMethod']]
    exp_metadata[['testType']] <- exp_metadata_mochi[['testType']]
    exp_metadata[['orderSubset']] <- exp_metadata_mochi[['orderSubset']]
    exp_metadata[['inferAdditiveTrait']] <- exp_metadata_mochi[['inferAdditiveTrait']]
  }

  #Create mochi directory (if doesn't already exist)
  exp_metadata[["mochi_path"]] <- file.path(exp_metadata[["tmp_path"]], exp_metadata[["analysisName"]])
  suppressWarnings(dir.create(exp_metadata[["mochi_path"]]))

  ### Pipeline stages
  ###########################

  ### Start pipeline tracking
  pipeline <- list()
  pipeline[['initial']] <- exp_metadata

  ### Stage 0: Infer additive trait using artificial neural network
  pipeline[['0_additivetrait']] <- mochi_stage_infer_additive_trait(
    mochi_meta = pipeline[['initial']],
    additivetrait_outpath = file.path(pipeline[['initial']][["tmp_path"]], exp_metadata[["analysisName"]], "0_additivetrait"),
    report_outpath = file.path(pipeline[['initial']][["tmp_path"]], exp_metadata[["analysisName"]], "0_additivetrait"))

  ### Stage 1: Determine combinatorial complete landscapes
  pipeline[['1_landscapes']] <- mochi_stage_combinatorial_complete_landscapes(
    mochi_meta = pipeline[['0_additivetrait']],
    landscapes_outpath = file.path(pipeline[['0_additivetrait']][["tmp_path"]], exp_metadata[["analysisName"]], "1_landscapes"),
    order_max = pipeline[['0_additivetrait']][["maxOrder"]],
    report_outpath = file.path(pipeline[['0_additivetrait']][["tmp_path"]], exp_metadata[["analysisName"]], "1_landscapes"))

  ### Stage 2: Obtain all epistatic terms, background relative and background averaged terms
  pipeline[['2_epistasis']] <- mochi_stage_higher_order_epistasis(
    mochi_meta = pipeline[['1_landscapes']],
    epistasis_outpath = file.path(pipeline[['1_landscapes']][["tmp_path"]], exp_metadata[["analysisName"]], "2_epistasis"),
    report_outpath = file.path(pipeline[['1_landscapes']][["tmp_path"]], exp_metadata[["analysisName"]], "2_epistasis"))

  ### Stage 3: Reconstruct fitness from individual terms of epistasis
  pipeline[['3_reconstruct']] <- mochi_stage_reconstruct_fitness(
    mochi_meta = pipeline[['2_epistasis']],
    reconstruct_outpath = file.path(pipeline[['2_epistasis']][["tmp_path"]], exp_metadata[["analysisName"]], "3_reconstruct"),
    report_outpath = file.path(pipeline[['2_epistasis']][["tmp_path"]], exp_metadata[["analysisName"]], "3_reconstruct"))

  ### Save workspace
  ###########################

  message("Done")

}
