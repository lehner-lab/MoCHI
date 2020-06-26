
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
  execute <- (dimsum_meta[["mochiStartStage"]] <= this_stage & (dimsum_meta[["mochiStopStage"]] == 0 | dimsum_meta[["mochiStopStage"]] >= this_stage))
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
  cclands_fitness_list <- mochi__add_fitness_to_landscapes(
    landscape_list = cclands_list,
    fitness_dt = fitness_dt,
    numCores = dimsum_meta[['numCores']])

  #Calculate all individual terms of epistasis
  cclands_epistasis_dt <- mochi__calculate_individual_epistasis_terms(
    input_list = cclands_fitness_list,
    genotype_key = genokey_dt,
    degreesFreedom = dimsum_meta[['numReplicates']]-1,
    test_type = dimsum_meta[['testType']],
    numCores = dimsum_meta[['numCores']])

  #Calculate background averaged terms of epistasis
  bckgavg_epistasis_dt <- mochi__calculate_background_averaged_epistasis_terms(
    input_dt = cclands_epistasis_dt,
    degreesFreedom = dimsum_meta[['numReplicates']]-1,
    test_type = dimsum_meta[['testType']],    
    FDR_threshold = dimsum_meta[['FDR_threshold']],
    numCores = dimsum_meta[['numCores']])

  #Save results
  save(cclands_fitness_list, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_complete_landscapes_fitness.RData')))
  save(cclands_epistasis_dt, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_individual_epistasis_terms.RData')))
  save(bckgavg_epistasis_dt, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_background_averaged_epistasis_terms.RData')))

  #Summary report and tables
  ###########################

  #Plot individual epistasis terms
  mochi_stage_higher_order_epistasis_report(
    dimsum_meta = dimsum_meta,
    report_outpath = dimsum_meta[["epistasis_path"]])

  #Get a summary of the combinatorial of complete landscapes and the number of backgrounds where each n combination of mutations are found
  cclands_summary = do.call("rbind", lapply(names(cclands_fitness_list), function(x){
    n_bckg <- plyr::count(cclands_fitness_list[[x]], c(1:3))[,"freq"]
    data.table(
      n = as.numeric(x), 
      n_comb=length(n_bckg),
      n_landscapes = dim(cclands_fitness_list[[x]])[1], 
      min_bckg = min(n_bckg), 
      median_bckg = median(n_bckg), 
      max_bckg = max(n_bckg))
  }))

  #Background averaged epistasis summary
  baepistasis_summary <- rbindlist(lapply(unique(bckgavg_epistasis_dt[,n]), function(x){
    data.table(
      n = x, 
      sig_global_ep = sum(bckgavg_epistasis_dt[n==x, global_qval_all < dimsum_meta[['FDR_threshold']]]), 
      n_comb = bckgavg_epistasis_dt[n == x,.N])
  }))
  print(baepistasis_summary)

  #Save summary tables
  save(cclands_summary, baepistasis_summary, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_epistasis_summary_tables.RData')))

}
