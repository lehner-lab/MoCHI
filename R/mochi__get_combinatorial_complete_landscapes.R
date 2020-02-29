
#' mochi__get_combinatorial_complete_landscapes
#'
#' Get all combinatorial complete landscapes of order nmut in all_variants_char.
#'
#' @param nmut an integer number of mutations to consider (default:1)
#' @param all_variants_char a character vector of variants each of identical character length (required)
#'
#' @return A data.table 
#' @export
#' @import data.table
mochi__get_combinatorial_complete_landscapes <- function(
  nmut = 1, 
  all_variants_char){

  #Initialise result list
  result_list <- list()

  #Check variant sequences of equal length
  if(length(unique(nchar(all_variants_char)))!=1){
    stop(paste0("Cannot compute combinatorial complete landscapes. Variant sequences of unequal length."), call. = FALSE)
  }

  #Check that nmut <= sequence length
  if(nchar(all_variants_char[1]) < nmut){
    return(NULL)
  }

  #All positions
  all_pos <- 1:nchar(all_variants_char[1])
  #Data table of variants
  var_dt <- data.table(var = all_variants_char)

  #Loop over all position combinations of length nmut
  for(this_pos in as.data.table(combn(all_pos, nmut))) {
    #Initialse list of results for this position combination
    result_list[[paste(this_pos, collapse=",")]] <- list()
    #Indices of variant background positions
    back_pos <- all_pos[!all_pos %in% this_pos]
    #Initialise mutant and background strings
    var_dt[, var_mut := ""]
    var_dt[, var_back := ""]
    
    #Construct separate mutant and background strings
    idx <- 0
    for(b in all_pos){
      idx <- idx + 1
      if(b %in% this_pos){
        var_dt[, var_mut := paste0(var_mut, substr(var, idx, idx))]
      }else{
        var_dt[, var_back := paste0(var_back, substr(var, idx, idx))]
      }
    }

    #Loop over all possible variant combinations of order nmut at these positions
    for(this_mut in var_dt[!grepl("0", var_mut),unique(var_mut)]){
      #Construct regular expression matching all possible combinations of these variants w.r.t. WT (coded as 0)
      this_regex <- paste0("[0", paste(unlist(strsplit(this_mut, "")), collapse="][0"), "]")
      #Subset to variants matching this pattern
      temp_dt <- var_dt[grepl(this_regex, var_mut),]
      #Subset to complete landscapes
      temp_table <- temp_dt[,table(var_back)]
      #Order by background and then mutant strings
      temp_dt <- temp_dt[var_back %in% names(temp_table)[temp_table==2^length(this_pos)],][order(var_back, var_mut),]

      #Construct list equivalent of data (each list item is the same variant combination in a different background)
      temp_list <- list()
      count <- 0
      for(i in unique(temp_dt[,var_mut])){
        count <- count + 1
        count_mut <- sum(unlist(strsplit(i, ""))!="0")
        temp_list[[count]] <- data.table(temp_dt[var_mut==i,var])
        names(temp_list[[count]]) <- paste("id", count_mut, count, sep = "_")
      }

      #Construct data.table equivalent of data (each row contains all combinations of a given variant in a different background)
      if(length(temp_list)!=0){
        result_dt <- do.call("cbind", temp_list)
        #Add metadata
        result_dt[, num_mut := nmut]
        result_dt[, pos_mut := paste(this_pos, collapse="-")]
        result_dt[, vars_mut := paste(unlist(strsplit(this_mut, "")), collapse="-")]
        #Reorder columns
        result_dt <- result_dt[,.SD,,.SDcols = c("num_mut", "pos_mut", "vars_mut", 
          names(result_dt)[!names(result_dt) %in% c("num_mut", "pos_mut", "vars_mut")])]
        #Save
        result_list[[paste(this_pos, collapse=",")]][[this_mut]] <- result_dt
      }
    }
    #Combine all mutation combinations at the same positions
    result_list[[paste(this_pos, collapse=",")]] <- do.call("rbind", result_list[[paste(this_pos, collapse=",")]])
  }
  #Combine all mutation combinations at different positions and return
  return(do.call("rbind", result_list))
}


