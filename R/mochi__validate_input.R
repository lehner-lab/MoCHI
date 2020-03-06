
#' mochi__validate_input
#'
#' Check validity and reformat input parameters.
#'
#' @param input_list a list of input arguments (required)
#'
#' @return an experiment metadata object
#' @export
#' @import data.table
mochi__validate_input <- function(
  input_list
  ){
  #Check all input parameter types correct
  for(input_arg in names(input_list)){
    #Integer arguments
    if("integer" %in% input_list[[input_arg]][[2]]){
      #Check if numeric
      if(!typeof(input_list[[input_arg]][[1]]) %in% c(input_list[[input_arg]][[2]], "double")){
        stop(paste0("Invalid type of argument '", input_arg, "' (", input_list[[input_arg]][[2]], ")"), call. = FALSE)
      }else{
        #Check if whole number (if not NULL)
        if(!is.null(input_list[[input_arg]][[1]])){
          if(input_list[[input_arg]][[1]]%%1!=0){
            stop(paste0("Invalid type of argument '", input_arg, "' (", input_list[[input_arg]][[2]], ")"), call. = FALSE)
          }
          #Convert to integer
          input_list[[input_arg]][[1]] <- as.integer(input_list[[input_arg]][[1]])
        }
      }
    }else{
      #Other arguments
      if(!typeof(input_list[[input_arg]][[1]]) %in% input_list[[input_arg]][[2]]){
        stop(paste0("Invalid type of argument '", input_arg, "' (", input_list[[input_arg]][[2]], ")"), call. = FALSE)
      }
    }
  }
  #Metadata object
  mochi_meta <- sapply(input_list, '[', 1)

  #Reformat directory paths and check if they exist (remove trailing "/" if present)
  for(input_arg in c("outputPath")){
    mochi_meta[[input_arg]] <- gsub("/$", "", mochi_meta[[input_arg]])
    #Check if directory paths exist
    if(!file.exists(mochi_meta[[input_arg]])){
      stop(paste0("Invalid '", input_arg, "' argument (directory not found)"), call. = FALSE)
    }
  }

  #Check mochiStartStage argument
  if(mochi_meta[["mochiStartStage"]]<=0){
    stop("Invalid 'mochiStartStage' argument. Only positive integers allowed (zero exclusive).", call. = FALSE)
  }

  #Check mochiStopStage argument
  if(mochi_meta[["mochiStopStage"]]<0){
    stop("Invalid 'mochiStopStage' argument. Only positive integers allowed (zero inclusive).", call. = FALSE)
  }

  #Check numCores argument
  if(mochi_meta[["numCores"]]<=0){
    stop("Invalid 'numCores' argument. Only positive integers allowed (zero exclusive).", call. = FALSE)
  }

  #Check maxOrder argument
  if(mochi_meta[["maxOrder"]]<=0){
    stop("Invalid 'maxOrder' argument. Only positive integers allowed (zero exclusive).", call. = FALSE)
  }

  #Check if either workspacePath or (inputFile, outputPath and projectName) specified
  if(!( !is.null(mochi_meta[["workspacePath"]]) | (!is.null(mochi_meta[["inputFile"]]) & !is.null(mochi_meta[["outputPath"]]) & !is.null(mochi_meta[["projectName"]])))) {
    stop(paste0("One or more mandatory arguments missing. If workspacePath not supplied, inputFile, outputPath and projectName need to be supplied. "), call. = FALSE)
  }

  #Check workspacePath exists
  if(!is.null(mochi_meta[["workspacePath"]])){
    if(!file.exists(mochi_meta[["workspacePath"]])){
      stop(paste0("Invalid '", "workspacePath", "' argument (file not found)"), call. = FALSE)
    }
    mochi_meta[["inputFile"]] <- NULL
    mochi_meta[["outputPath"]] <- NULL
    mochi_meta[["projectName"]] <- NULL    
  }

  #Check if inputFile specified
  if(!is.null(mochi_meta[["inputFile"]])){
    #Load input file
    if(!file.exists(mochi_meta[["inputFile"]])){
      stop(paste0("Invalid '", "inputFile", "' argument (file not found)"), call. = FALSE)
    }
    #Load file
    input_dt <- fread(mochi_meta[["inputFile"]], nrows = 1)
    #Check if mandatory columns present
    mandatory_cols <- c("WT", "fitness", "sigma")
    if(sum(unlist(lapply(mandatory_cols, "%in%", colnames(input_dt)))==FALSE)!=0){
      stop(paste0("One or more mandatory columns missing from inputFile ('", paste(mandatory_cols, collapse = "', '"), "')"), call. = FALSE)
    }
    if("nt_seq" %in% colnames(input_dt)){
      mochi_meta[["sequenceType"]] <- "noncoding"
    }else if(sum(c("aa_seq", "STOP") %in% colnames(input_dt))==2){
      mochi_meta[["sequenceType"]] <- "coding"
    }else{
      stop(paste0("One or more mandatory columns missing from inputFile ('nt_seq' or 'aa_seq' and 'STOP')"), call. = FALSE)
    }
  }

  #Return
  return(mochi_meta)
}
