
#' mochi__v_matrix
#'
#' Construct V matrix.
#'
#' @param maxOrder maximum order (required)
#'
#' @return A matrix
#' @export
#' @import data.table
mochi__v_matrix <- function(
  maxOrder
  ){  
  V.Matrix <- matrix(
    data = 1, 
    nrow = 1, 
    ncol = 1)
  for (j in 1:maxOrder) {
    Zero.Matrix <- matrix(
      data = 0,
      nrow = nrow(V.Matrix),
      ncol = ncol(V.Matrix))
    Top.Row <- cbind(0.5*V.Matrix, Zero.Matrix)
    Bottom.Row <- cbind(Zero.Matrix, -1*V.Matrix)
    V.Matrix <- rbind(Top.Row, Bottom.Row)
  }
  return(V.Matrix)
}