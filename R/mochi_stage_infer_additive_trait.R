
#' mochi_stage_infer_additive_trait
#'
#' Infer additive trait using artifical neural network.
#'
#' @param mochi_meta an experiment metadata object (required)
#' @param additivetrait_outpath output path for saved objects (required)
#' @param report whether or not to generate fitness summary plots (default: TRUE)
#' @param report_outpath fitness report output path
#' @param save_workspace whether or not to save the current workspace (default: TRUE)
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_infer_additive_trait <- function(
  mochi_meta,
  additivetrait_outpath,
  report = TRUE,
  report_outpath = NULL,
  save_workspace = TRUE
  ){
  #Whether or not to execute the system command
  this_stage <- 0
  execute <- (mochi_meta[["inferAdditiveTrait"]] & mochi_meta[["mochiStartStage"]] <= this_stage & (mochi_meta[["mochiStopStage"]] == 0 | mochi_meta[["mochiStopStage"]] >= this_stage))

  #Save current workspace for debugging purposes
  if(save_workspace){mochi__save_metadata(mochi_meta = mochi_meta, n = 2)}

  #Additive trait path
  mochi_meta[["additivetrait_path"]] <- additivetrait_outpath

  #Create/overwrite additivetrait directory (if executed)
  additivetrait_outpath <- gsub("/$", "", additivetrait_outpath)
  mochi__create_dir(additivetrait_outpath, execute = execute, message = "MoCHI STAGE 0: INFER ADDITIVE TRAIT") 

  if(!execute){return(mochi_meta)}

  #Sequence type
  sequence_type <- "nucleotide"
  if(mochi_meta[["sequenceType"]]=="coding"){
    sequence_type <- "aminoacid"
  }

  ### Load variant data
  ###########################

  #load variant data from RData file
  if(!file.exists(file.path(mochi_meta[["fitness_path"]], paste0(mochi_meta[["projectName"]], '_fitness_replicates.RData')))){
    stop(paste0("Cannot determine combinatorial complete landscapes. Fitness file not found."), call. = FALSE)
  }else{
    load(file.path(mochi_meta[["fitness_path"]], paste0(mochi_meta[["projectName"]], '_fitness_replicates.RData')))
  }

  #Remove sequences with STOP codons or silent mutations if amino acid sequence, check for duplicates
  if(sequence_type == "aminoacid"){
    all_variants <- all_variants[STOP==F & !(Nham_aa==0 & WT==F)]
    if(all_variants[duplicated(aa_seq),.N]!=0){
      stop(paste0("Cannot infer additive trait. Duplicate genotypes not permitted."), call. = FALSE)
    }
  }else{
    if(all_variants[duplicated(nt_seq),.N]!=0){
      stop(paste0("Cannot infer additive trait. Duplicate genotypes not permitted."), call. = FALSE)
    }
  }

  ### Save input variant data for neural network
  ###########################

  #Add variant sequence
  if(sequence_type == "aminoacid"){
    all_variants[, var_seq := aa_seq]
  }else{
    all_variants[, var_seq := nt_seq]
  }

  #Write observed fitness table
  observed_fitness_file <- file.path(additivetrait_outpath, "observed_fitness.txt")
  write.table(
    all_variants[,.(fitness, var_seq)], 
    file = observed_fitness_file, 
    quote = F, sep = "\t", row.names = F)

  ### Infer additive trait using artifical neural network
  ###########################

  #Run python script on command-line
  system(paste0(
    system.file("python", "mochi__infer_additive_trait.py", package = "MoCHI"),
    " -i ",
    observed_fitness_file,
    " -o ",
    additivetrait_outpath))

  ### Append predicted fitness and additive trait to variant data and save
  ###########################

  all_variants_predicted <- fread(file.path(additivetrait_outpath, "predicted_fitness.txt"))

  #Check that input and output files are similarly ordered
  if(sum(all_variants_predicted[,seq]!=all_variants[,var_seq])!=0){
    stop(paste0("Cannot infer additive trait. Error running mochi__infer_additive_trait.py."), call. = FALSE)
  }

  #Append results to variant data
  all_variants[, fitness_predicted := all_variants_predicted[,predicted_fitness]]
  all_variants[, additive_trait := all_variants_predicted[,additive_trait]]

  #Replace fitness with residual fitness
  all_variants[, fitness_observed := fitness]
  all_variants[, fitness := fitness_observed-fitness_predicted]

  #Save variant data
  all_variants <- all_variants[,.SD,,.SDcols = names(all_variants)[names(all_variants)!="var_seq"]]
  save(
    all_variants, 
    file = file.path(additivetrait_outpath, paste0(mochi_meta[["projectName"]], '_fitness_replicates.RData')))

  return(mochi_meta)
}
