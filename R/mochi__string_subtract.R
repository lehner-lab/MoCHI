
#' mochi__string_subtract
#'
#' Subract string2 from string1 (where strings are equal length and string2 is a zero-masked version of string1).
#'
#' @param string1 string 1 (required)
#' @param string2 string 2 (required)
#'
#' @return A string
#' @export
mochi__string_subtract <- function(
  string1, 
  string2
  ){
  strv1 <- unlist(strsplit(string1, ""))
  strv2 <- unlist(strsplit(string2, ""))
  strvr <- rep("0", nchar(string1))
  strvr[which(strv1!=strv2)] <- strv1[which(strv1!=strv2)]
  return(paste0(strvr, collapse = ""))
}
