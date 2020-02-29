
#' mochi_stage_higher_order_epistasis_report
#'
#' Generate higher order epistasis summary plots.
#'
#' @param dimsum_meta an experiment metadata object (required)
#' @param report_outpath final report output path (required)
#'
#' @return Nothing
#' @export
#' @import data.table
mochi_stage_higher_order_epistasis_report <- function(
  dimsum_meta,
  report_outpath
  ){
  #Create report directory (if doesn't already exist)
  report_outpath <- gsub("/$", "", report_outpath)
  suppressWarnings(dir.create(report_outpath))

  #Load data
  load(file.path(dimsum_meta[["epistasis_path"]], paste0(dimsum_meta[["projectName"]], '_epistasis_terms.RData')))

  #Plot effect of all single mutants in different backgrounds - first order
  EpGlobal[, perc_positive := positive_fdr10/n_bckg*100]
  EpGlobal[, perc_negative := negative_fdr10/n_bckg*100]
  EpGlobal[, perc_neutral := neutral_fdr10/n_bckg*100]
  EpGlobal[, Mutant_sig := paste0(Mutant, c("", " (*)")[as.numeric(global_state!="Neutral")+1])]

  #Plot effect of all single mutants in different backgrounds - first order
  for(this_order in 1:3){
    plot_dt <- reshape2::melt(EpGlobal[n == this_order,.SD,.SDcols = c("Mutant_sig", "perc_positive", "perc_negative", "perc_neutral")], id = "Mutant_sig")
    plot_dt[,Mutant_sig := factor(Mutant_sig, levels = as.character(EpGlobal[n==this_order,Mutant_sig][order(EpGlobal[n==this_order,(perc_negative+1)/(perc_positive+1)])]))]
    plot_dt[,variable := factor(variable, levels = c("perc_negative", "perc_neutral", "perc_positive"))]
    if(this_order==1){
      fill_cols <- c("blue", "lightgrey", "red")
    }else{
      fill_cols <- c("green", "lightgrey", "orange")
    }
    names(fill_cols) <- c("perc_negative", "perc_neutral", "perc_positive")
    d <- ggplot2::ggplot(plot_dt,ggplot2::aes(x = Mutant_sig, y = value, fill = variable)) +
      ggplot2::geom_bar(stat = "identity") +
      # ggplot2::ylab("Correlation with starting fitness") +
      ggplot2::scale_fill_manual(values = fill_cols) +
      ggplot2::theme_bw() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, hjust=1))  # ggplot2::geom_vline(xintercept = 0) +
    ggplot2::ggsave(file.path(report_outpath, paste0("fitness_effect_order", this_order, "_sigperc_backgrounds.pdf")), d, width = 7, height = 5, useDingbats=FALSE)
  }

}

