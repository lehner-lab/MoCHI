
load("JD_Phylogeny_tR-R-CCU_fitness_replicates.RData")

output.table <- all_variants[,c("fitness", "nt_seq")]
names(output.table)[2] <- "aa_seq"
write.table(x = output.table,
            file = "Julia.txt",
            quote = FALSE,
            row.names = FALSE,
            col.names = TRUE,
            sep = "\t")
