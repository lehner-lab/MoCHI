library(data.table)

highest_order = 7

load("mochi_output/S7_1D/tmp/epistasis/S7_1D_epistasis_terms.RData")
Epistasis_1D <- EpGlobal
rm(CD_ep, EpGlobal, L_CD_fit)

load("mochi_output/S7_RAW/tmp/epistasis/S7_RAW_epistasis_terms.RData")
Epistasis_Raw <- EpGlobal
rm(CD_ep, EpGlobal, L_CD_fit)



# first table: how many interactions that were significant in the
# raw data are still significant after removing global epistasis

vector_signficant_terms_raw <- c()
vector_preserved_signficant_terms_1d <- c()
amount_of_interactions_1d <- c()


for (i in 1:highest_order) {
  
  terms_raw <- Epistasis_Raw[n == i, global_state]
  terms_1d <- Epistasis_1D[n == i, global_state]
  
  
  significant_terms_raw <- sum(terms_raw != "Neutral")
  significant_terms_1d <- sum((terms_raw != "Neutral") & (terms_1d != "Neutral"))
  
  vector_signficant_terms_raw <- c(vector_signficant_terms_raw,
                                   significant_terms_raw)
  
  vector_preserved_signficant_terms_1d <- c(vector_preserved_signficant_terms_1d,
                                            significant_terms_1d)
  
  significant_terms_1d <- sum((terms_1d != "Neutral"))
  
  amount_of_interactions_1d <- c(amount_of_interactions_1d,
                                 significant_terms_1d)

}



load("mochi_output/S7_RAW/tmp/epistasis/S7_RAW_epistasis_summary_tables.RData")


Table.To.Save <- data.frame(n = 1:highest_order,
                            total_possible_combinations = baepistasis_summary$n_comb,
                            number_interactions_raw = vector_signficant_terms_raw,
                            number_interactions_1d = amount_of_interactions_1d,
                            number_preserved_interactions_1d = vector_preserved_signficant_terms_1d) 




write.table(x = Table.To.Save,
            file = "epistasis_summary.txt",
            quote = F,
            sep = "\t",
            row.names = F,
            col.names = T)
