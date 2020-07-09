


Mochi.Table <- read.table(file = "data/mochi_table.txt",
                          header = TRUE,
                          stringsAsFactors = FALSE,
                          sep = "\t")

Corrected.Fitness.Scores <- read.table(file = "002_LM_Predictions.txt",
                                       header = TRUE,
                                       stringsAsFactors = TRUE,
                                       sep = "\t")

Mochi.Table$fitness <- Corrected.Fitness.Scores$observed_fitness - Corrected.Fitness.Scores$predicted_fitness

write.table(x = Mochi.Table,
            file = "data/mochi_table_LM.txt",
            quote = F,
            sep = "\t",
            row.names = F,
            col.names = T)
