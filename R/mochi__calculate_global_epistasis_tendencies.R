
#' mochi__calculate_global_epistasis_tendencies
#'
#' Calculate general tendecies of epistasis at any order.
#'
#' @param input_dt data.table as provided by calculate_individual_epistatis_terms (required)
#' @param numCores number of available CPU cores (default:1)
#'
#' @return A data.table with background averaged epistasis terms, direction and significance
#' @export
#' @import data.table
mochi__calculate_global_epistasis_tendencies <- function(
  input_dt,
  numCores = 1
  ){
  L_Mutants <- tapply(input_dt[,Mutant], input_dt[,n], unique)
  result_list <- parallel::mclapply(as.list(unique(input_dt[,n])),function(x){
    temp <- input_dt[n == x,]
    DS_list <- lapply(L_Mutants[[as.character(x)]], function(y){
      n_bckg <- sum(temp[,Mutant] == y)
      mean_ep_term <- temp[Mutant==y, mean(ep_term)]
      global_SE <- temp[Mutant==y, (1/n_bckg) * sqrt(sum(ep_term_SE^2))]
      positive_fdr10 = temp[Mutant==y,sum(ep_term > 0 & qval_all < 0.1)]
      negative_fdr10 = temp[Mutant==y,sum(ep_term < 0 & qval_all < 0.1)]
      data.table(
        Mutant = y, 
        mean_ep_term = mean_ep_term, 
        n_bckg = n_bckg,
        positive_fdr10 = positive_fdr10, 
        negative_fdr10 = negative_fdr10,
        neutral_fdr10 = n_bckg - (positive_fdr10 + negative_fdr10),
        global_SE = global_SE, 
        global_pval = mochi__ttest(
          av = mean_ep_term, 
          se = global_SE, 
          df = 5))
    })
    DS <- rbindlist(DS_list)
    DS[, global_qval_order := p.adjust(global_pval, method = "fdr")]
    DS[, n := x]
    DS[,.SD,,.SDcols = c("n", names(DS)[names(DS)!="n"])]
  },mc.cores = numCores)
  #Merge
  merged_list <- data.table::rbindlist(result_list)
  #Global adjusted p-values
  merged_list[, global_qval_all := p.adjust(global_pval, method = "fdr")]
  #Global state
  merged_list[, global_state := "Neutral"]
  merged_list[global_qval_all < 0.1 & mean_ep_term < 0, global_state := "Negative"]
  merged_list[global_qval_all < 0.1 & mean_ep_term >= 0, global_state := "Positive"]
  merged_list[, n := factor(n)]
  return(merged_list)
}
