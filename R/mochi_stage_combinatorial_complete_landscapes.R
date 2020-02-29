
#' mochi_stage_combinatorial_complete_landscapes
#'
#' Determine combinatorial complete landscapes.
#'
#' @param dimsum_meta an experiment metadata object (required)
#' @param epistasis_outpath output path for saved objects (required)
#' @param order_max an integer number of maximum mutations to consider (default:1)
#' @param report whether or not to generate fitness summary plots (default: TRUE)
#' @param report_outpath fitness report output path
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_combinatorial_complete_landscapes <- function(
  dimsum_meta,
  epistasis_outpath,
  order_max = 1,
  report = TRUE,
  report_outpath = NULL,
  save_workspace = TRUE
  ){
  #Whether or not to execute the system command
  this_stage <- 1
  execute <- (dimsum_meta[["HOEStartStage"]] <= this_stage & (dimsum_meta[["HOEStopStage"]] == 0 | dimsum_meta[["HOEStopStage"]] >= this_stage))
  #Epistasis paths
  dimsum_meta[["epistasis_path"]] <- file.path(epistasis_outpath)

  #Create/overwrite fitness directory (if executed)
  epistasis_outpath <- gsub("/$", "", epistasis_outpath)
  mochi__create_dir(epistasis_outpath, execute = execute, message = "MoCHI STAGE: DETERMINE COMBINATORIAL COMPLETE LANDSCAPES", overwrite_dir = FALSE) 

  if(!execute){return()}

  #Sequence type
  sequence_type <- "nucleotide"
  if(dimsum_meta[["sequenceType"]]=="coding"){
    sequence_type <- "aminoacid"
  }

  ###########################
  ### Variant fitness >> combinatorial complete landscapes
  ###########################

  ### Load variant data
  ###########################

  #load variant data from RData file
  load(file.path(dimsum_meta[["fitness_path"]], paste0(dimsum_meta[["projectName"]], '_fitness_replicates.RData')))

  #Remove sequences with STOP codons or silent mutations if amino acid sequence
  if(sequence_type == "aminoacid"){
    all_variants <- all_variants[STOP==F & !(Nham_aa==0 & WT==F)]
  }

  #Encode genotypes
  P_list <- mochi__encode_genotypes(
    input_dt = all_variants, 
    sequenceType = sequence_type)
  P <- P_list[["P"]]
  P_single <- P_list[["P_single"]]

  #Save genotype fitness
  save(P, P_single, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_genotypes_fitness.RData')))

  #Get all combinatorial complete landscapes of all orders
  L_DS <- mochi__get_all_combinatorial_complete_landscapes(
    order_max = order_max,
    all_variants_char = P[,var],
    numCores = dimsum_meta[['numCores']]-1)

  #Save all combinatorial complete landscapes of all orders
  save(L_DS, file = file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_complete_landscapes.RData')))

  #Get a summary of the combinatorial of complete landscapes and the number of backgrounds where each n combination of mutations are found
  DS_summary = do.call("rbind", lapply(names(L_DS), function(x){
    n_bckg <- plyr::count(L_DS[[x]], c(1:3))[,"freq"]
    data.table(
      n = as.numeric(x), 
      n_comb=length(n_bckg),
      n_landscapes = dim(L_DS[[x]])[1], 
      min_bckg = min(n_bckg), 
      median_bckg = median(n_bckg), 
      max_bckg = max(n_bckg))
  }))
  print(DS_summary)

}
