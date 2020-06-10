
#' mochi__ttest
#'
#' TDist wrapper.
#'
#' @param av sample mean (required)
#' @param se sample standard error (required)
#' @param degreesFreedom an integer degrees of freedom (default:5)
#' @param mu population mean (default:0)
#'
#' @return A data.table 
#' @export
mochi__ttest <- function(
  av, 
  se, 
  degreesFreedom = 5, 
  mu = 0){
  tstat <- (av - mu)/se
  # Using T-dist
  pval <- 2*pt(abs(tstat), degreesFreedom, lower = FALSE)
  return(pval)
}
