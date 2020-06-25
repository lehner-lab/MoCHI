
#' mochi__reconstruct_fitness
#'
#' Reconstruct fitness from individual terms of epistasis.
#'
#' @param input_list list of combinatoral landscape data.tables with corresponding fitness and error as provided by mochi__add_fitness_to_landscapes (required)
#' @param single_mut_dt data.table of single substitution variants (required)
#' @param order_subset comma-separated list of (integer) orders of epistatic terms to retain for fitness reconstruction (required)
#' @param numCores number of available CPU cores (default:1)
#'
#' @return A data.table 
#' @export
#' @import data.table
mochi__reconstruct_fitness <- function(
  input_list,
  single_mut_dt,
  order_subset,
  numCores = 1
  ){
  
  #Reformat order subset
  order_subset <- as.integer(strsplit(order_subset, ",")[[1]])

  #List to convert from variant id (single character variant string where WT=0) to id (comma-separated mutation string) 
  P_var2id <- mochi__construct_var2id_list(input_dt = single_mut_dt)

  # Calculate all individual terms of epistasis
  output_list <- parallel::mclapply(as.list(as.numeric(names(input_list))),function(x){
    l_cd_fit_temp <- data.table::copy(input_list[[as.character(x)]])

    #Calculate epistatic terms of order x
    g_mat <- mochi__g_matrix(x)
    ep_terms <- as.data.table(t(g_mat%*%t(as.matrix(l_cd_fit_temp[,.SD,,.SDcols = grep("^fit_", names(l_cd_fit_temp))]))))
    names(ep_terms) <- gsub("fit_", "ep_", names(l_cd_fit_temp)[grep("^fit_", names(l_cd_fit_temp))])

    #Reconstruct fitness
    fit_rec <- ep_terms[,rowSums(.SD),,.SDcols = grep(paste0("^ep_(", paste(order_subset, collapse = "|"), ")_"), names(ep_terms))]

    #Add variant information
    l_cd_fit_temp[, fit_rec := fit_rec]
    l_cd_fit_temp[, id_bckg := unlist(P_var2id[unlist(.SD)]),,.SDcols = grep("^id_0_", names(l_cd_fit_temp))]
    l_cd_fit_temp[, id_mut := unlist(P_var2id[unlist(.SD)]),,.SDcols = grep(paste0("^id_", x, "_"), names(l_cd_fit_temp))]
    l_cd_fit_temp[, Mutant := mapply(mochi__string_subtract, .SD[[2]], .SD[[1]]),,.SDcols = range(grep("^id_.*_", names(l_cd_fit_temp)))]
    l_cd_fit_temp[, Mutant := unlist(P_var2id[Mutant])]
    l_cd_fit_temp
  },mc.cores = numCores)

  #Rename list elements
  names(output_list) <- names(input_list)

  #Merge list retaining only desired columns
  merged_list <- data.table::rbindlist(lapply(names(output_list), function(x){
    output_list[[x]][, n := as.numeric(x)]
    output_list[[x]][,.SD,,.SDcols = grep("(^id_[1-9])|(^fit_[1-9])|(^se_)", names(output_list[[x]]), invert = T)]
  }))

  return(merged_list)
}
