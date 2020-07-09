
load("data/Pokusaev2019_S7_fitness_replicates.RData")

y <- all_variants$fitness

Variance.Explained <- c()

for (i in 1:100){
  print(i)
  X <- c()
  for (j in 1:nrow(all_variants)){
    set.seed(i)
    x <- rnorm(n = 1,
               mean = all_variants$fitness[j],
               sd = all_variants$sigma[j]*sqrt(2))
    X <- c(X, x)
  }
  # X <- rnorm(n = nrow(all_variants),
  #            mean = all_variants$fitness,
  #            sd = all_variants$sigma)
  
  # calculate variance explained
  Mean.Y <- mean(y,na.rm = T)
  SStotal <- sum((y - Mean.Y)^2, na.rm = T)
  
  linear.model <- lm(y~X)
  SSresidual <- sum(linear.model$residuals^2, na.rm = T)

  # append to vector
  Variance.Explained <- c(Variance.Explained,
                          1 - (SSresidual/SStotal))
  
}

Explainable.Variance <- mean(Variance.Explained,
                             na.rm = TRUE)

write.table(x = data.frame(X = Explainable.Variance),
            file = "003_explainable_variance.txt",
            row.names = FALSE,
            col.names = FALSE)
