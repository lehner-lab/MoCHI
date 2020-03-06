
#' mochi__encode_genotypes
#'
#' Encode genotypes mapping id (comma-separated mutation string) to var (single character variant string where WT=0) for variable positions only.
#'
#' @param input_dt a data.table of variants (required)
#' @param sequenceType sequence type: one of nucleotide/aminoacid (default:nucleotide)
#'
#' @return A named list of two data.tables
#' @export
#' @import data.table
mochi__encode_genotypes <- function(
  input_dt, 
  sequenceType = "nucleotide"
  ){

  #Set and check sequence type
  if(sequenceType == "nucleotide"){
    input_dt[, seq := .SD,,.SDcols = "nt_seq"]
  }else if(sequenceType == "aminoacid"){
    input_dt[, seq := .SD,,.SDcols = "aa_seq"]
  }else{
    stop(paste0("Cannot encode genotypes. Invalid 'sequenceType' argument. Only nucleotide/aminoacid allowed."), call. = FALSE)
  }

  #Add mutant codes
  for(i in 1:nchar(input_dt[WT==T,seq])){
    input_dt[, paste0("Mut_pos", i) := paste0(i, substr(seq, i, i))]
  }

  #Mutated positions
  mut_colnames <- names(input_dt)[grep("^Mut_", names(input_dt))]
  var_num <- sapply(lapply(as.list(as.data.frame(input_dt[,.SD,,.SDcols = mut_colnames])), table), length)
  mut_colnames <- mut_colnames[var_num!=1]
  input_dt[,Mut := apply(.SD, 1, paste, collapse=","),,.SDcols = mut_colnames]

  #id column
  P <- data.table::copy(input_dt)
  for(i in 1:nchar(input_dt[WT==T,seq])){
    P[, paste0("Mut_pos", i) := toupper(paste0(substr(P[WT==T,seq], i, i), .SD[[1]])),,.SDcols = paste0("Mut_pos", i)]
    P[substr(seq, i, i)==substr(P[WT==T,seq], i, i), paste0("Mut_pos", i) := NA]
  }
  P[, id := gsub(",NA|NA,", "", apply(.SD, 1, paste0, collapse = ",")),,.SDcols = mut_colnames]
  P[WT==T, id := "-0-"]

  #var column
  P_single <- list()
  for(i in 1:nchar(input_dt[WT==T,seq])){
    P[, paste0("Mut_seq", i) := substr(seq, i, i)]
    P[substr(seq, i, i)==substr(P[WT==T,seq], i, i), paste0("Mut_seq", i) := "0"]
  
    if(paste0("Mut_pos", i) %in% mut_colnames){
      P_single[[i]] <- data.table(id = unique(unlist(P[,.SD,,.SDcols = paste0("Mut_pos", i)])))
      rpad <- length(mut_colnames)-which(mut_colnames==paste0("Mut_pos", i))+1
      P_single[[i]][, var := unique(unlist(P[,.SD,,.SDcols = paste0("Mut_seq", i)]))]
      P_single[[i]][, var := stringr::str_pad(var, rpad, side = "right", pad = "0")]
      P_single[[i]][, var := stringr::str_pad(var, length(mut_colnames), side = "left", pad = "0")]
      P_single[[i]] <- P_single[[i]][!is.na(id)]
    }
  }
  P[, var := apply(.SD, 1, paste0, collapse = ""),,.SDcols = gsub("_pos", "_seq", mut_colnames)]
  P_single <- do.call("rbind", P_single)

  #SE column
  P[, SE := sigma]

  #Final data.table
  P <- P[,.SD,,.SDcols = c("fitness", "SE", "id", "var")]
  return(list(P = P, P_single = P_single))
}
