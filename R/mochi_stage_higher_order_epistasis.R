
#' mochi_stage_higher_order_epistasis
#'
#' Obtain all epistatic terms, background relative and background averaged terms.
#'
#' @param mochi_meta an experiment metadata object (required)
#' @param epistasis_outpath output path for saved objects (required)
#' @param report whether or not to generate fitness summary plots (default: TRUE)
#' @param report_outpath fitness report output path
#' @param save_workspace whether or not to save the current workspace (default: TRUE)
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_higher_order_epistasis <- function(
  mochi_meta,
  epistasis_outpath,
  report = TRUE,
  report_outpath = NULL,
  save_workspace = TRUE
  ){
  #Whether or not to execute the system command
  this_stage <- 2
  execute <- (mochi_meta[["mochiStartStage"]] <= this_stage & (mochi_meta[["mochiStopStage"]] == 0 | mochi_meta[["mochiStopStage"]] >= this_stage))

  #Save current workspace for debugging purposes
  if(save_workspace){mochi__save_metadata(mochi_meta = mochi_meta, n = 2)}

  #Epistasis path
  mochi_meta[["epistasis_path"]] <- file.path(epistasis_outpath)

  #Create/overwrite fitness directory (if executed)
  epistasis_outpath <- gsub("/$", "", epistasis_outpath)
  mochi__create_dir(epistasis_outpath, execute = execute, message = "MoCHI STAGE 2: CALCULATE HIGHER ORDER EPISTASIS TERMS", overwrite_dir = FALSE) 

  if(!execute){return(mochi_meta)}

  #Sequence type
  sequence_type <- "nucleotide"
  if(mochi_meta[["sequenceType"]]=="coding"){
    sequence_type <- "aminoacid"
  }

  ###########################
  ### Combinatorial complete landscapes >> global epistasis tendencies
  ###########################

  #load variant data and landscapes from RData file
  load(file.path(mochi_meta[["landscapes_path"]], paste0(mochi_meta[["projectName"]], '_genotypes_fitness.RData')))
  load(file.path(mochi_meta[["landscapes_path"]], paste0(mochi_meta[["projectName"]], '_complete_landscapes.RData')))

  #Add fitness to landscapes
  cclands_fitness_list <- mochi__add_fitness_to_landscapes(
    landscape_list = cclands_list,
    fitness_dt = fitness_dt,
    numCores = mochi_meta[['numCores']])

  #Calculate all individual terms of epistasis
  cclands_epistasis_dt <- mochi__calculate_individual_epistasis_terms(
    input_list = cclands_fitness_list,
    genotype_key = genokey_dt,
    degreesFreedom = mochi_meta[['numReplicates']]-1,
    testType = mochi_meta[['testType']],
    adjustmentMethod = mochi_meta[['adjustmentMethod']],
    numCores = mochi_meta[['numCores']])

  #Calculate background averaged terms of epistasis
  bckgavg_epistasis_dt <- mochi__calculate_background_averaged_epistasis_terms(
    input_dt = cclands_epistasis_dt,
    degreesFreedom = mochi_meta[['numReplicates']]-1,
    testType = mochi_meta[['testType']],    
    significanceThreshold = mochi_meta[['significanceThreshold']],
    adjustmentMethod = mochi_meta[['adjustmentMethod']],
    numCores = mochi_meta[['numCores']])

  #Save results
  save(cclands_fitness_list, file = file.path(epistasis_outpath, paste0(mochi_meta[["projectName"]], '_complete_landscapes_fitness.RData')))
  save(cclands_epistasis_dt, file = file.path(epistasis_outpath, paste0(mochi_meta[["projectName"]], '_individual_epistasis_terms.RData')))
  save(bckgavg_epistasis_dt, file = file.path(epistasis_outpath, paste0(mochi_meta[["projectName"]], '_background_averaged_epistasis_terms.RData')))

  #Summary report and tables
  ###########################

  #Plot individual epistasis terms
  mochi__higher_order_epistasis_report(
    mochi_meta = mochi_meta,
    report_outpath = epistasis_outpath)

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
      sig_global_ep = sum(bckgavg_epistasis_dt[n==x, global_qval_all < mochi_meta[['significanceThreshold']]]), 
      n_comb = bckgavg_epistasis_dt[n == x,.N])
  }))
  print(baepistasis_summary)

  #Save summary tables
  save(cclands_summary, baepistasis_summary, file = file.path(epistasis_outpath, paste0(mochi_meta[["projectName"]], '_epistasis_summary_tables.RData')))

  return(mochi_meta)
}
