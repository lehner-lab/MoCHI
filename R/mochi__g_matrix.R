
#' mochi__g_matrix
#'
#' Construct G matrix.
#'
#' @param maxOrder maximum order (required)
#'
#' @return A matrix
#' @export
#' @import data.table
mochi__g_matrix <- function(
  maxOrder
  ){
  G.Matrix <- matrix(
    data = 1, 
    nrow = 1, 
    ncol = 1)
  for (j in 1:maxOrder) {
    Top.Row <- cbind(G.Matrix, 0*G.Matrix)
    Bottom.Row <- cbind(-1*G.Matrix, G.Matrix)
    G.Matrix <- rbind(Top.Row, Bottom.Row)
  }
  return(G.Matrix)
}