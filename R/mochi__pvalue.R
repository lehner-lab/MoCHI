
#' mochi__pvalue
#'
#' Obtain P-value from z test or t test
#'
#' @param av sample mean (required)
#' @param se sample standard error (required)
#' @param degreesFreedom an integer degrees of freedom (default:5)
#' @param mu population mean (default:0)
#' @param testType type of test: either "ztest" or "ttest" (default:"ztest")
#'
#' @return P-value
#' @export
mochi__pvalue <- function(
  av, 
  se, 
  degreesFreedom = 5, 
  mu = 0,
  testType = "ztest"){

	#Perform z test
	if(testType=="ztest"){
	  zscore <- (av - mu)/se
	  pval <- 2*pnorm(abs(zscore), lower.tail = FALSE)
	  return(pval)
	}

	#Perform t test
	if(testType=="ttest"){
	  tstat <- (av - mu)/se
	  pval <- 2*pt(abs(tstat), degreesFreedom, lower = FALSE)
	  return(pval)
	}

}
