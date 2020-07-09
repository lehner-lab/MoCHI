
Model.Table <- read.table("001_1D_Predictions.txt",
                          header = TRUE,
                          stringsAsFactors = FALSE,
                          quote = "")

y <- Model.Table$observed_fitness
X <- Model.Table$predicted_fitness

# calculate variance explained
Mean.Y <- mean(y,na.rm = T)
SStotal <- sum((y - Mean.Y)^2, na.rm = T)

SSresidual <- sum((y - X)^2, na.rm = T)

Explained.Variance <- 1 - (SSresidual/SStotal)


write.table(x = data.frame(X = Explained.Variance),
            file = "004_ann_explained_variance.txt",
            row.names = FALSE,
            col.names = FALSE)






Model.Table <- read.table("002_LM_Predictions.txt",
                          header = TRUE,
                          stringsAsFactors = FALSE,
                          quote = "")

y <- Model.Table$observed_fitness
X <- Model.Table$predicted_fitness

# calculate variance explained
Mean.Y <- mean(y,na.rm = T)
SStotal <- sum((y - Mean.Y)^2, na.rm = T)

SSresidual <- sum((y - X)^2, na.rm = T)

Explained.Variance <- 1 - (SSresidual/SStotal)


write.table(x = data.frame(X = Explained.Variance),
            file = "004_lm_explained_variance.txt",
            row.names = FALSE,
            col.names = FALSE)
