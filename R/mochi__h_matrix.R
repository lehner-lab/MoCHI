
#' mochi__h_matrix
#'
#' Construct H matrix.
#'
#' @param maxOrder maximum order (required)
#'
#' @return A matrix
#' @export
#' @import data.table
mochi__h_matrix <- function(
  maxOrder
  ){
  H.Matrix <- matrix(
    data = 1, 
    nrow = 1, 
    ncol = 1)
  for (j in 1:maxOrder) {
    Top.Row <- cbind(H.Matrix, H.Matrix)
    Bottom.Row <- cbind(H.Matrix, -1*H.Matrix)
    H.Matrix <- rbind(Top.Row, Bottom.Row)
  }
  return(H.Matrix)
}