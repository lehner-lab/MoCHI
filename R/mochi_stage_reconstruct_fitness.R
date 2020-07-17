
#' mochi_stage_reconstruct_fitness
#'
#' Reconstruct fitness from individual terms of epistasis.
#'
#' @param mochi_meta an experiment metadata object (required)
#' @param reconstruct_outpath output path for saved objects (required)
#' @param report whether or not to generate fitness summary plots (default: TRUE)
#' @param report_outpath fitness report output path
#' @param save_workspace whether or not to save the current workspace (default: TRUE)
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_reconstruct_fitness <- function(
  mochi_meta,
  reconstruct_outpath,
  report = TRUE,
  report_outpath = NULL,
  save_workspace = TRUE
  ){
  #Whether or not to execute the system command
  this_stage <- 3
  execute <- (mochi_meta[["mochiStartStage"]] <= this_stage & (mochi_meta[["mochiStopStage"]] == 0 | mochi_meta[["mochiStopStage"]] >= this_stage))

  #Save current workspace for debugging purposes
  if(save_workspace){mochi__save_metadata(mochi_meta = mochi_meta, n = 2)}

  #Epistasis paths
  mochi_meta[["reconstruct_path"]] <- file.path(reconstruct_outpath)

  #Create/overwrite fitness directory (if executed)
  reconstruct_outpath <- gsub("/$", "", reconstruct_outpath)
  mochi__create_dir(reconstruct_outpath, execute = execute, message = "MoCHI STAGE 3: RECONSTRUCT FITNESS FROM INDIVIDUAL EPISTASIS TERMS", overwrite_dir = FALSE) 

  if(!execute){return(mochi_meta)}

  ###########################
  ### Reconstruct fitness from individual terms of epistasis
  ###########################

  #load variant data and landscapes from RData file
  load(file.path(mochi_meta[["landscapes_path"]], paste0(mochi_meta[["projectName"]], '_genotypes_fitness.RData')))
  load(file.path(mochi_meta[["epistasis_path"]], paste0(mochi_meta[["projectName"]], '_complete_landscapes_fitness.RData')))

  #Reconstruct fitness from individual terms of epistasis
  cclands_recfitness_dt <- mochi__reconstruct_fitness(
    input_list = cclands_fitness_list,
    genotype_key = genokey_dt,
    order_subset = mochi_meta[['orderSubset']],
    numCores = mochi_meta[['numCores']])

  #Save results
  save(cclands_recfitness_dt, file = file.path(reconstruct_outpath, paste0(mochi_meta[["projectName"]], '_reconstructed_fitness_orders', gsub(",", "_", mochi_meta[['orderSubset']]),'.RData')))

  return(mochi_meta)
}
