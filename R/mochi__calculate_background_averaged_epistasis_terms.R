
#' mochi__calculate_background_averaged_epistasis_terms
#'
#' Calculate general tendecies of epistasis at any order.
#'
#' @param input_dt data.table as provided by calculate_individual_epistatis_terms (required)
#' @param degreesFreedom an integer degrees of freedom (default:5)
#' @param test_type type of test: either "ztest" or "ttest" (default:"ztest")
#' @param FDR_threshold FDR threshold for significant specific and background averaged terms (default:0.1)
#' @param numCores number of available CPU cores (default:1)
#'
#' @return A data.table with background averaged epistasis terms, direction and significance
#' @export
#' @import data.table
mochi__calculate_background_averaged_epistasis_terms <- function(
  input_dt,
  degreesFreedom = 5,
  test_type = "ztest",
  FDR_threshold = 0.1,
  numCores = 1
  ){
  L_Mutants <- tapply(input_dt[,Mutant], input_dt[,n], unique)
  result_list <- parallel::mclapply(as.list(unique(input_dt[,n])),function(x){
    temp <- input_dt[n == x,]
    DS_list <- lapply(L_Mutants[[as.character(x)]], function(y){
      n_bckg <- sum(temp[,Mutant] == y)
      mean_ep_term <- temp[Mutant==y, mean(ep_term)]
      global_SE <- temp[Mutant==y, (1/n_bckg) * sqrt(sum(ep_term_SE^2))]
      positive_fdr = temp[Mutant==y,sum(ep_term > 0 & qval_all < FDR_threshold)]
      negative_fdr = temp[Mutant==y,sum(ep_term < 0 & qval_all < FDR_threshold)]
      data.table(
        Mutant = y, 
        mean_ep_term = mean_ep_term, 
        n_bckg = n_bckg,
        positive_fdr = positive_fdr, 
        negative_fdr = negative_fdr,
        neutral_fdr = n_bckg - (positive_fdr + negative_fdr),
        global_SE = global_SE, 
        global_pval = mochi__pvalue(
          av = mean_ep_term, 
          se = global_SE, 
          degreesFreedom = degreesFreedom,
          test_type = test_type))
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
  merged_list[global_qval_all < FDR_threshold & mean_ep_term < 0, global_state := "Negative"]
  merged_list[global_qval_all < FDR_threshold & mean_ep_term >= 0, global_state := "Positive"]
  merged_list[, n := factor(n)]
  return(merged_list)
}
