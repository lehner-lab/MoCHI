
#' mochi__construct_var2id_list
#'
#' Get mapping of ids to variants for all possible combinations of single mutants supplied in data.table with id and var columns.
#'
#' @param input_dt data.table of single substitution variants (required)
#'
#' @return A data.table 
#' @export
#' @import data.table
mochi__construct_var2id_list <- function(
  input_dt
  ){
  
  #Construct list mapping all possible variants to ids 
  var_mat <- do.call("rbind", lapply(strsplit(input_dt[,var], ""), unlist))
  rownames(var_mat) <- input_dt[,id]
  temp <- data.table(count = 1:prod(sapply(lapply(as.list(as.data.frame(var_mat)), unique), length)))
  rep_each <- 1
  for(pos in 1:nchar(input_dt[1,var])){
    temp[, paste0("Mut_var", pos) := rep(unique(var_mat[,pos])[which(unique(var_mat[,pos]) != "0")], each = rep_each)]
    rep_each <- rep_each * length(unique(var_mat[,pos])[which(unique(var_mat[,pos]) != "0")])
    var_dict <- rownames(var_mat)[var_mat[,pos]!="0"]
    names(var_dict) <- var_mat[,pos][var_mat[,pos]!="0"]
    temp[, paste0("Mut_id", pos) := var_dict[unlist(.SD)],,.SDcols = paste0("Mut_var", pos)]
    temp[is.na(get(paste0("Mut_id", pos))), paste0("Mut_id", pos) := NA]
  }
  temp[, var := apply(.SD, 1, paste0, collapse = ""),,.SDcols = grep("Mut_var", names(temp))]
  temp[, id := apply(.SD, 1, paste0, collapse = ","),,.SDcols = grep("Mut_id", names(temp))]
  temp[, id := gsub(",NA|NA,", "", id)]
  temp[id == "NA", id := "-0-"]
  var2id <- as.list(temp[, id])
  names(var2id) <- temp[, var]
  return(var2id)
}
