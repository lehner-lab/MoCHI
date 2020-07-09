

load("data/Pokusaev2019_S7_fitness_replicates.RData")

Mochi.Table <- all_variants[,c("WT",
                               "fitness",
                               "sigma",
                               "aa_seq",
                               "STOP",
                               "Nham_aa")]



write.table(x = Mochi.Table,
            file = "data/mochi_table.txt",
            quote = F,
            sep = "\t",
            row.names = F,
            col.names = T)
