
#' mochi_stage_higher_order_epistasis
#'
#' Obtain all epistatic terms, background relative and background averaged terms.
#'
#' @param dimsum_meta an experiment metadata object (required)
#' @param epistasis_outpath output path for saved objects (required)
#' @param report whether or not to generate fitness summary plots (default: TRUE)
#' @param report_outpath fitness report output path
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_higher_order_epistasis <- function(
  dimsum_meta,
  epistasis_outpath,
  report = TRUE,
  report_outpath = NULL,
  save_workspace = TRUE
  ){
  #Whether or not to execute the system command
  this_stage <- 2
  execute <- (dimsum_meta[["HOEStartStage"]] <= this_stage & (dimsum_meta[["HOEStopStage"]] == 0 | dimsum_meta[["HOEStopStage"]] >= this_stage))
  #Epistasis paths
  dimsum_meta[["epistasis_path"]] <- file.path(epistasis_outpath)

  #Create/overwrite fitness directory (if executed)
  epistasis_outpath <- gsub("/$", "", epistasis_outpath)
  mochi__create_dir(epistasis_outpath, execute = execute, message = "MoCHI STAGE: CALCULATE HIGHER ORDER EPISTASIS TERMS", overwrite_dir = FALSE) 

  if(!execute){return()}

  #Sequence type
  sequence_type <- "nucleotide"
  if(dimsum_meta[["sequenceType"]]=="coding"){
    sequence_type <- "aminoacid"
  }

  ###########################
  ### Combinatorial complete landscapes >> global epistasis tendencies
  ###########################

  #load variant data and landscapes from RData file
  load(file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_genotypes_fitness.RData')))
  load(file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_complete_landscapes.RData')))

  #Add fitness to landscapes
  L_CD_fit <- mochi__add_fitness_to_landscapes(
    landscape_list = L_DS,
    fitness_dt = P,
    numCores = dimsum_meta[['numCores']]-1)

  #Calculate all individual terms of epistasis
  CD_ep <- mochi__calculate_individual_epistatis_terms(
    input_list = L_CD_fit,
    single_mut_dt = P_single,
    numCores = dimsum_meta[['numCores']]-1)

  #General tendecies of epistasis at any order
  EpGlobal <- mochi__calculate_global_epistasis_tendencies(
    input_dt = CD_ep,
    numCores = dimsum_meta[['numCores']]-1)

  #Save results
  save(L_CD_fit, CD_ep, EpGlobal, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_epistasis_terms.RData')))

  #Summary report and tables
  ###########################

  #Plot individual epistasis terms
  mochi_stage_higher_order_epistasis_report(
    dimsum_meta = dimsum_meta,
    report_outpath = dimsum_meta[["epistasis_path"]])

  #Get a summary of the combinatorial of complete landscapes and the number of backgrounds where each n combination of mutations are found
  DS_summary = do.call("rbind", lapply(names(L_CD_fit), function(x){
    n_bckg <- plyr::count(L_CD_fit[[x]], c(1:3))[,"freq"]
    data.table(
      n = as.numeric(x), 
      n_comb=length(n_bckg),
      n_landscapes = dim(L_CD_fit[[x]])[1], 
      min_bckg = min(n_bckg), 
      median_bckg = median(n_bckg), 
      max_bckg = max(n_bckg))
  }))

  #Background averaged epistasis summary
  SummaryEpGlobal <- rbindlist(lapply(unique(EpGlobal[,n]), function(x){
    data.table(
      n = x, 
      sig_global_ep = sum(EpGlobal[n==x, global_qval_all < 0.1]), 
      n_comb = EpGlobal[n == x,.N])
  }))
  print(SummaryEpGlobal)

  #Save summary tables
  save(DS_summary, SummaryEpGlobal, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_epistasis_summary_tables.RData')))

}
