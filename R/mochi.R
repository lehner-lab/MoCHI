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
#'
#' @return Nothing
#' @export
#' @import data.table
mochi <- function(
  outputPath,
  projectName,
  startStage=1,
  stopStage=0,
  numCores=1,
  maxOrder=2
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
    "workspacePath" = list(arg_list[["workspacePath"]], c("character", "NULL")), #file exists (if not NULL) -- checked in dimsum__validate_input
    "inputFile" = list(arg_list[["inputFile"]], c("character", "NULL")), #file exists (if not NULL) -- checked in dimsum__validate_input
    "outputPath" = list(arg_list[["outputPath"]], c("character", "NULL")), #directory exists (if not NULL) -- checked in mochi__validate_input
    "projectName" = list(arg_list[["projectName"]], c("character", "NULL")), #character string (if not NULL) -- checked in mochi__validate_input
    "mochiStartStage" = list(arg_list[["startStage"]], c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "mochiStopStage" = list(arg_list[["stopStage"]], c("integer")), #positive integer (zero inclusive) -- checked in mochi__validate_input
    "numCores" = list(arg_list[["numCores"]], c("integer")), #strictly positive integer -- checked in mochi__validate_input
    "maxOrder" = list(arg_list[["maxOrder"]], c("integer")) #strictly positive integer -- checked in mochi__validate_input
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
    #Load data
    load(exp_metadata[["workspacePath"]])
    if(!'7_fitness' %in% names(pipeline)){
      stop(paste0("Invalid '", "workspacePath", "' argument (stage 7)"), call. = FALSE)
    }
    exp_metadata <- pipeline[['7_fitness']]
    exp_metadata[['mochiStartStage']] <- arg_list[["startStage"]]
    exp_metadata[['mochiStopStage']] <- arg_list[["stopStage"]]
    exp_metadata[['numCores']] <- arg_list[["numCores"]]
    exp_metadata[['maxOrder']] <- arg_list[["maxOrder"]]
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
