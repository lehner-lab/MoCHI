
#' mochi__add_fitness_to_landscapes
#'
#' Add fitness and error to landscapes.
#'
#' @param landscape_list list of landscape data (required)
#' @param fitness_dt data.table of variant fitness and error (required)
#' @param numCores number of available CPU cores (default:1)
#'
#' @return A list of data.tables
#' @export
#' @import data.table
mochi__add_fitness_to_landscapes <- function(
  landscape_list,
  fitness_dt,
  numCores = 1
  ){

  #List to convert from variant id (single character variant string where WT=0) to fitness and SE
  var2fit <- as.list(as.data.frame(t(fitness_dt[,.(fitness, SE)])))
  names(var2fit) <- fitness_dt[,var]

  #All orders of landscapes
  all_orders <- as.numeric(names(landscape_list))

  #Append fitness and error
  result_list <- parallel::mclapply(as.list(all_orders),function(x){
    this_order <- all_orders[x]
    comb <- rep(0:this_order, choose(this_order, 0:this_order))
    fit_names <- paste("fit_", comb, "_", 1:length(comb), sep="")
    se_names <- paste("se_", comb, "_", 1:length(comb), sep="")
    CL <- landscape_list[[as.character(x)]]
    #For all variant columns
    for(i in 4:ncol(CL)){
      CL[, gsub("id", "fit", names(CL)[i]) := sapply(var2fit[unlist(.SD)], '[', 1),,.SDcols = names(CL)[i]]
      CL[, gsub("id", "se", names(CL)[i]) := sapply(var2fit[unlist(.SD)], '[', 2),,.SDcols = names(CL)[i]]
    }
    CL
  },mc.cores = numCores)

  #Rename list elements
  names(result_list) <- as.character(all_orders)
  return(result_list)
}
