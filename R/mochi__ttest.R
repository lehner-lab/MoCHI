
#' mochi__ttest
#'
#' TDist wrapper.
#'
#' @param av sample mean (required)
#' @param se sample standard error (required)
#' @param df an integer degrees of freedom (default:5)
#' @param mu population mean (default:0)
#'
#' @return A data.table 
#' @export
mochi__ttest <- function(
  av, 
  se, 
  df = 5, 
  mu = 0){
  tstat <- (av - mu)/se
  # Using T-dist
  pval <- 2*pt(abs(tstat), df, lower = FALSE)
  return(pval)
}
