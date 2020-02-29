
#' mochi__create_dir
#'
#' Create results folder for mochi pipeline stage.
#'
#' @param mochi_dir directory path string (required)
#' @param execute whether or not the system command will be executed (required)
#' @param message message string (optional, default: NULL i.e. no message displayed)
#' @param overwrite_dir delete directory if already exists (optional, default: TRUE)
#'
#' @return Nothing
#' @export
mochi__create_dir <- function(
  mochi_dir, 
  execute, 
  message = NULL, 
  overwrite_dir = TRUE){
  if(!execute){
    if(!is.null(message)){
      message(paste("\n\n\n*******", message, "(not executed)", "*******\n\n\n"))
    }
    return()
  }
  if(!is.null(message)){
    message(paste("\n\n\n*******", message, "*******\n\n\n"))
  }
  if(dir.exists(mochi_dir) & overwrite_dir){
    unlink(mochi_dir, recursive = TRUE)
  }
  dir.create(mochi_dir, showWarnings = FALSE)
}
