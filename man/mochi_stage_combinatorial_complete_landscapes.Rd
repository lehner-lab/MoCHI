% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mochi_stage_combinatorial_complete_landscapes.R
\name{mochi_stage_combinatorial_complete_landscapes}
\alias{mochi_stage_combinatorial_complete_landscapes}
\title{mochi_stage_combinatorial_complete_landscapes}
\usage{
mochi_stage_combinatorial_complete_landscapes(mochi_meta,
  landscapes_outpath, order_max = 1, report = TRUE,
  report_outpath = NULL, save_workspace = TRUE)
}
\arguments{
\item{mochi_meta}{an experiment metadata object (required)}

\item{landscapes_outpath}{output path for saved objects (required)}

\item{order_max}{an integer number of maximum mutations to consider (default:1)}

\item{report}{whether or not to generate fitness summary plots (default: TRUE)}

\item{report_outpath}{fitness report output path}

\item{save_workspace}{whether or not to save the current workspace (default: TRUE)}
}
\value{
Nothing
}
\description{
Determine combinatorial complete landscapes.
}
