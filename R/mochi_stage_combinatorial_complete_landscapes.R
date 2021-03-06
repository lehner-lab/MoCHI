
#' mochi_stage_combinatorial_complete_landscapes
#'
#' Determine combinatorial complete landscapes.
#'
#' @param mochi_meta an experiment metadata object (required)
#' @param landscapes_outpath output path for saved objects (required)
#' @param order_max an integer number of maximum mutations to consider (default:1)
#' @param report whether or not to generate fitness summary plots (default: TRUE)
#' @param report_outpath fitness report output path
#' @param save_workspace whether or not to save the current workspace (default: TRUE)
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_combinatorial_complete_landscapes <- function(
  mochi_meta,
  landscapes_outpath,
  order_max = 1,
  report = TRUE,
  report_outpath = NULL,
  save_workspace = TRUE
  ){
  #Whether or not to execute the system command
  this_stage <- 1
  execute <- (mochi_meta[["mochiStartStage"]] <= this_stage & (mochi_meta[["mochiStopStage"]] == 0 | mochi_meta[["mochiStopStage"]] >= this_stage))

  #Save current workspace for debugging purposes
  if(save_workspace){mochi__save_metadata(mochi_meta = mochi_meta, n = 2)}

  #Landscapes path
  mochi_meta[["landscapes_path"]] <- landscapes_outpath

  #Create/overwrite fitness directory (if executed)
  landscapes_outpath <- gsub("/$", "", landscapes_outpath)
  mochi__create_dir(landscapes_outpath, execute = execute, message = "MoCHI STAGE 1: DETERMINE COMBINATORIAL COMPLETE LANDSCAPES", overwrite_dir = FALSE) 

  if(!execute){return(mochi_meta)}

  #Sequence type
  sequence_type <- "nucleotide"
  if(mochi_meta[["sequenceType"]]=="coding"){
    sequence_type <- "aminoacid"
  }

  ###########################
  ### Variant fitness >> combinatorial complete landscapes
  ###########################

  ### Load variant data
  ###########################

  #Set variant data path
  variant_data_path <- file.path(mochi_meta[["fitness_path"]], paste0(mochi_meta[["projectName"]], '_fitness_replicates.RData'))
  if(mochi_meta[["inferAdditiveTrait"]]){
    variant_data_path <- file.path(mochi_meta[["additivetrait_path"]], paste0(mochi_meta[["projectName"]], '_fitness_replicates.RData'))
  }

  #load variant data from RData file
  if(!file.exists(variant_data_path)){
    stop(paste0("Cannot determine combinatorial complete landscapes. Fitness file not found."), call. = FALSE)
  }else{
    load(variant_data_path)
  }

  #Remove sequences with STOP codons or silent mutations if amino acid sequence, check for duplicates
  if(sequence_type == "aminoacid"){
    all_variants <- all_variants[STOP==F & !(Nham_aa==0 & WT==F)]
    if(all_variants[duplicated(aa_seq),.N]!=0){
      stop(paste0("Cannot determine combinatorial complete landscapes. Duplicate genotypes not permitted."), call. = FALSE)
    }
  }else{
    if(all_variants[duplicated(nt_seq),.N]!=0){
      stop(paste0("Cannot determine combinatorial complete landscapes. Duplicate genotypes not permitted."), call. = FALSE)
    }
  }

  #Encode genotypes
  fitness_list <- mochi__encode_genotypes(
    input_dt = all_variants, 
    sequenceType = sequence_type)
  fitness_dt <- fitness_list[["all"]]
  genokey_dt <- fitness_list[["genotype_key"]]

  #Save genotype fitness
  save(fitness_dt, genokey_dt, file = file.path(landscapes_outpath, paste0(mochi_meta[["projectName"]], '_genotypes_fitness.RData')))

  #Get all combinatorial complete landscapes of all orders
  cclands_list <- mochi__get_all_combinatorial_complete_landscapes(
    order_max = order_max,
    all_variants_char = fitness_dt[,var],
    numCores = mochi_meta[['numCores']])

  #Save all combinatorial complete landscapes of all orders
  save(cclands_list, file = file.path(landscapes_outpath, paste0(mochi_meta[["projectName"]], '_complete_landscapes.RData')))

  #Get a summary of the combinatorial of complete landscapes and the number of backgrounds where each n combination of mutations are found
  cclands_summary = do.call("rbind", lapply(names(cclands_list), function(x){
    n_bckg <- plyr::count(cclands_list[[x]], c(1:3))[,"freq"]
    data.table(
      n = as.numeric(x), 
      n_comb=length(n_bckg),
      n_landscapes = dim(cclands_list[[x]])[1], 
      min_bckg = min(n_bckg), 
      median_bckg = median(n_bckg), 
      max_bckg = max(n_bckg))
  }))
  print(cclands_summary)

  return(mochi_meta)
}
