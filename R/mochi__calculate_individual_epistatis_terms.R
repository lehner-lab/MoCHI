
#' mochi__calculate_individual_epistatis_terms
#'
#' Calculate all individual terms of epistasis.
#'
#' @param input_list list of combinatoral landscape data.tables with corresponding fitness and error as provided by mochi__add_fitness_to_landscapes (required)
#' @param single_mut_dt data.table of single substitution variants (required)
#' @param numCores number of available CPU cores (default:1)
#'
#' @return A data.table with individual epistatic terms
#' @export
#' @import data.table
mochi__calculate_individual_epistatis_terms <- function(
  input_list,
  single_mut_dt,
  numCores = 1
  ){
  
  #List to convert from variant id (single character variant string where WT=0) to id (comma-separated mutation string) 
  P_var2id <- mochi__construct_var2id_list(input_dt = single_mut_dt)

  # Calculate all individual terms of epistasis
  output_list <- parallel::mclapply(as.list(as.numeric(names(input_list))),function(x){
    l_cd_fit_temp <- data.table::copy(input_list[[as.character(x)]])

    orders = x:0
    sums_v = rep(T, length(orders))
    if (orders[1] %% 2 == 0) {
      sums_v[!(orders) %% 2 == 0] = F
    } else{
      sums_v[(orders) %% 2 == 0] = F
    }

    temp = do.call("cbind", lapply(orders, function(y){
      order_regex = paste0("fit_", y, "_")
      if (y == x | y == 0) {
        l_cd_fit_temp[,.SD,, .SDcols = grep(order_regex, names(l_cd_fit_temp))]
      } else {
        rowSums(l_cd_fit_temp[,.SD,, .SDcols = grep(order_regex, names(l_cd_fit_temp))])
      }
    }))
    
    if (sum(sums_v) > 1) {
      ep_term = rowSums(temp[,.SD,,.SDcols = which(sums_v)])
    } else {
      ep_term = temp[,.SD,,.SDcols = which(sums_v)]
    }
    
    if (sum(!sums_v) > 1) {
      ep_term = ep_term - rowSums(temp[,.SD,,.SDcols = which(!sums_v)])
    } else {
      ep_term = ep_term - temp[,.SD,,.SDcols = which(!sums_v)]
    }

    l_cd_fit_temp[, ep_term := ep_term]
    l_cd_fit_temp[, ep_term_SE := sqrt(rowSums(.SD^2)),,.SDcols = grep("^se_", names(l_cd_fit_temp))]
    l_cd_fit_temp[, pval := mochi__ttest(
      av = ep_term, 
      se = ep_term_SE, 
      mu=0, 
      df=5)]
    l_cd_fit_temp[, qval_order := p.adjust(pval, method = "fdr")]
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
  merged_list[, qval_all := p.adjust(pval, method = "fdr")]

  return(merged_list)
}
