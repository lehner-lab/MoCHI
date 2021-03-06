
#' mochi__calculate_individual_epistasis_terms
#'
#' Calculate all individual terms of epistasis.
#'
#' @param input_list list of combinatoral landscape data.tables with corresponding fitness and error as provided by mochi__add_fitness_to_landscapes (required)
#' @param genotype_key data.table with genotype codes of single substitution variants (required)
#' @param degreesFreedom an integer degrees of freedom (default:5)
#' @param testType type of test: either "ztest" or "ttest" (default:"ztest")
#' @param adjustmentMethod Method for adjusting P-values for multiple comparisons; any method in p.adjust.methods is permitted (default:'fdr')
#' @param numCores number of available CPU cores (default:1)
#'
#' @return A data.table with individual epistatic terms
#' @export
#' @import data.table
mochi__calculate_individual_epistasis_terms <- function(
  input_list,
  genotype_key,
  degreesFreedom = 5,
  testType = "ztest",
  adjustmentMethod = "fdr",
  numCores = 1
  ){
  
  #List to convert from variant id (single character variant string where WT=0) to id (comma-separated mutation string) 
  P_var2id <- mochi__construct_var2id_list(input_dt = genotype_key)

  # Calculate all individual terms of epistasis
  output_list <- parallel::mclapply(as.list(as.numeric(names(input_list))),function(x){
    l_cd_fit_temp <- data.table::copy(input_list[[as.character(x)]])

    #Calculate epistatic terms of order x
    g_mat <- mochi__g_matrix(x)
    ep_term <- as.data.table(as.matrix(l_cd_fit_temp[,.SD,,.SDcols = grep("^fit_", names(l_cd_fit_temp))])%*%as.matrix(g_mat[dim(g_mat)[1],]))

    #Calculate significance of epistatic terms
    l_cd_fit_temp[, ep_term := ep_term]
    l_cd_fit_temp[, ep_term_SE := sqrt(rowSums(.SD^2)),,.SDcols = grep("^se_", names(l_cd_fit_temp))]
    l_cd_fit_temp[, pval := mochi__pvalue(
      av = ep_term, 
      se = ep_term_SE, 
      degreesFreedom = degreesFreedom,
      testType = testType)]
    l_cd_fit_temp[, qval_order := p.adjust(pval, method = adjustmentMethod)]
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
  merged_list[, qval_all := p.adjust(pval, method = adjustmentMethod)]

  return(merged_list)
}
