
#' mochi_stage_reconstruct_fitness
#'
#' Reconstruct fitness from individual terms of epistasis.
#'
#' @param dimsum_meta an experiment metadata object (required)
#' @param epistasis_outpath output path for saved objects (required)
#' @param report whether or not to generate fitness summary plots (default: TRUE)
#' @param report_outpath fitness report output path
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_reconstruct_fitness <- function(
  dimsum_meta,
  epistasis_outpath,
  report = TRUE,
  report_outpath = NULL,
  save_workspace = TRUE
  ){
  #Whether or not to execute the system command
  this_stage <- 3
  execute <- (dimsum_meta[["mochiStartStage"]] <= this_stage & (dimsum_meta[["mochiStopStage"]] == 0 | dimsum_meta[["mochiStopStage"]] >= this_stage))
  #Epistasis paths
  dimsum_meta[["epistasis_path"]] <- file.path(epistasis_outpath)

  #Create/overwrite fitness directory (if executed)
  epistasis_outpath <- gsub("/$", "", epistasis_outpath)
  mochi__create_dir(epistasis_outpath, execute = execute, message = "MoCHI STAGE: RECONSTRUCT FITNESS FROM INDIVIDUAL EPISTASIS TERMS", overwrite_dir = FALSE) 

  if(!execute){return()}

  ###########################
  ### Reconstruct fitness from individual terms of epistasis
  ###########################

  #load variant data and landscapes from RData file
  load(file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_genotypes_fitness.RData')))
  load(file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_complete_landscapes_fitness.RData')))

  #Reconstruct fitness from individual terms of epistasis
  cclands_recfitness_dt <- mochi__reconstruct_fitness(
    input_list = cclands_fitness_list,
    genotype_key = genokey_dt,
    order_subset = dimsum_meta[['orderSubset']],
    numCores = dimsum_meta[['numCores']])

  #Save results
  save(cclands_recfitness_dt, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_reconstructed_fitness_orders', gsub(",", "_", dimsum_meta[['orderSubset']]),'.RData')))

}
