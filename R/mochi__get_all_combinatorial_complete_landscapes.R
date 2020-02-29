
#' mochi__get_all_combinatorial_complete_landscapes
#'
#' Get all combinatorial complete landscapes of all orders in all_variants_char.
#'
#' @param order_max an integer number of maximum mutations to consider (default:1)
#' @param all_variants_char a character vector of variants each of identical length (required)
#' @param numCores number of available CPU cores (default:1)
#'
#' @return A list of data.tables
#' @export
#' @import data.table
mochi__get_all_combinatorial_complete_landscapes <- function(
  order_max = 1,
  all_variants_char,
  numCores = 1
  ){

  all_landscapes <- parallel::mclapply(as.list(1:order_max),function(co){
    mochi__get_combinatorial_complete_landscapes(nmut = co, all_variants_char = all_variants_char)
  },mc.cores = numCores)
  
  #Set list names
  names(all_landscapes) <- as.character(1:order_max)

  for(i in names(all_landscapes)){
    if(is.null(all_landscapes[[i]])){
      message(paste0("No combinatorial complete landscapes of order ", i, " found"))
    }
  }

  #Remove null entries
  all_landscapes <- all_landscapes[!sapply(all_landscapes, is.null)]

  return(all_landscapes)
}
