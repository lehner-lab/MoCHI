

load("data/JD_Phylogeny_tR-R-CCU_fitness_replicates.RData")

Mochi.Table <- all_variants[,c("WT",
                               "fitness",
                               "sigma",
                               "nt_seq",
                               "Nham_nt")]



write.table(x = Mochi.Table,
            file = "data/mochi_table.txt",
            quote = F,
            sep = "\t",
            row.names = F,
            col.names = T)
