##################################################
#############     Simulation Study    ############
#### Functional Neural Networks ##################
##################################################

###### MAIN CODE #######

##### Libraries #####
source("FNN.R")

#############################################################
# 1 - Identity
#############################################################

### Set.Seed
set.seed(1994)
use_session_with_seed(
  1994,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(t) function
beta_coef = runif(5, min = 0, max = 2)

# Generating alpha
alpha = runif(1, 0, 1)

# Initializing
RMSE_fnn <- c()
RMSE_lm <- c()
fnn_time <- c()
lm_time <- c()

# Running loop
for (u in 1:250) {
  
  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*exp(a) + sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = composite_approximator_other(response_func1,
                                        a = 0,
                                        b = 1,
                                        n = 500,
                                        x_obs = simSmooth$fd$coefs[,i], 
                                        beta = beta_coef[1]) +
      composite_approximator_other(response_func2,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[2]) +
      composite_approximator_other(response_func3,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[3]) +
      composite_approximator_other(response_func4,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[4]) +
      composite_approximator_other(response_func5,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[5]) +
      alpha
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  
  # Setting up grid
  start_time <- Sys.time()
  l = 2^(-4:10)
  
  # Running functional linear model
  func_basis = fregre.basis.cv(sim_fd, 
                               ystar, 
                               type.basis = "fourier",
                               lambda=l, 
                               type.CV = GCV.S, 
                               par.CV = list(trim=0.15))
  end_time <- Sys.time()
  lm_time[u] = end_time - start_time
  
  # Pulling out the coefficients
  coefficients_lm = func_basis$fregre.basis$coefficients
  
  # Setting up tensor for fnn
  sim_data_fnn = array(dim = c(5, 300, 1))
  sim_data_fnn[,,1] = sim_fd$coefs
  
  # Now running functional neural network
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  start_time <- Sys.time()
  fnn_sim = FNN(resp = ystar, 
                func_cov = sim_data_fnn, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 3,
                neurons_per_layer = c(16, 16, 16),
                activations_in_layers = c("relu", "linear", "linear"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  end_time <- Sys.time()
  
  # Getting time
  fnn_time[u] = end_time - start_time
  
  # Getting the FNC
  coefficients_fnn = rowMeans(get_weights(fnn_sim$model)[[1]])
  
  # IMSE - FNN
  result_fnn = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_fnn, lower = 0, upper = 1)
  
  # IMSE - LM
  result_lm = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_lm[,1], lower = 0, upper = 1)
  
  #######################################################
  
  # Saving RMSE
  RMSE_fnn[u] <- sqrt(result_fnn[[1]])
  RMSE_lm[u] <- sqrt(result_lm[[1]])
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
  # Printing progress
  print(paste0("Simulation: ", u, " = Done"))
}

# Summary statistics
fnn_mean1 = mean(RMSE_fnn)
lm_mean1 = mean(RMSE_lm)

fnn_sd1 = sd(RMSE_fnn)/sqrt(250)
lm_sd1 = sd(RMSE_lm)/sqrt(250)

# Times
fnn_time_sim1 = sum(fnn_time)
lm_time_sim1 = sum(lm_time)

# Putting together for boxplot
sim_result1 <- data.frame(FNN = RMSE_fnn, LM = RMSE_lm)

# Boxplot
sqrt_RMSE1 <- data.frame(sim_result1)

# Creating boxplots
plot1 = ggplot(stack(sqrt_RMSE1), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkblue") + 
  theme_bw() + 
  xlab("Model\n (A)") +
  ylab("IMSE") +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

# Running paired t-test
  
# Selecting data sets
FNN_ttest = RMSE_fnn
FLM_ttest = RMSE_lm

# Calculating difference
d = FNN_ttest - FLM_ttest

# Mean difference
mean_d = mean(d)

# SE
se_d = sd(d)/sqrt(length(FNN_ttest))

# T value
T_value = mean_d/se_d

# df
df_val = length(FNN_ttest) - 1

# p-value
p_value_sim1 = matrix(nrow = 1, ncol = 4)
rownames(p_value_sim1) = c("FNN v. FLM")
colnames(p_value_sim1) = c("P Value", "T Value", "Lower Bound", "Upper Upper Bound")
p_value_sim1[, 1] = pt(abs(T_value), df_val, lower.tail = F)
p_value_sim1[, 2] = T_value
p_value_sim1[, 3] = mean_d - 1.96*se_d
p_value_sim1[, 4] = mean_d + 1.96*se_d


#############################################################
# 2 - Exponential
#############################################################

### Set.Seed
set.seed(1994)
use_session_with_seed(
  1994,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(t) function
beta_coef = runif(5, min = 0, max = 2)

# Generating alpha
alpha = runif(1, 0, 1)

# Initializing
RMSE_fnn <- c()
RMSE_lm <- c()
fnn_time <- c()
lm_time <- c()

# Running loop
for (u in 1:250) {
  
  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = exp(composite_approximator_other(response_func1,
                                            a = 0,
                                            b = 1,
                                            n = 500,
                                            x_obs = simSmooth$fd$coefs[,i], 
                                            beta = beta_coef[1]) +
                 composite_approximator_other(response_func2,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[2]) +
                 composite_approximator_other(response_func3,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[3]) +
                 composite_approximator_other(response_func4,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[4]) +
                 composite_approximator_other(response_func5,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[5]) +
                 alpha)
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  
  # Setting up grid
  start_time <- Sys.time()
  l = 2^(-4:10)
  
  # Running functional linear model
  func_basis = fregre.basis.cv(sim_fd, 
                               ystar, 
                               type.basis = "fourier",
                               lambda=l, 
                               type.CV = GCV.S, 
                               par.CV = list(trim=0.15))
  end_time <- Sys.time()
  lm_time[u] = end_time - start_time
  
  # Pulling out the coefficients
  coefficients_lm = func_basis$fregre.basis$coefficients
  
  # Setting up tensor for fnn
  sim_data_fnn = array(dim = c(5, 300, 1))
  sim_data_fnn[,,1] = sim_fd$coefs
  
  # Now running functional neural network
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  start_time <- Sys.time()
  fnn_sim = FNN(resp = ystar, 
                func_cov = sim_data_fnn, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 3,
                neurons_per_layer = c(16, 16, 16),
                activations_in_layers = c("relu", "linear", "linear"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  end_time <- Sys.time()
  
  # Getting time
  fnn_time[u] = end_time - start_time
  
  # Getting the FNC
  coefficients_fnn = rowMeans(get_weights(fnn_sim$model)[[1]])
  
  # IMSE - FNN
  result_fnn = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_fnn, lower = 0, upper = 1)
  
  # IMSE - LM
  result_lm = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_lm[,1], lower = 0, upper = 1)
  
  #######################################################
  
  # Saving RMSE
  RMSE_fnn[u] <- sqrt(result_fnn[[1]])
  RMSE_lm[u] <- sqrt(result_lm[[1]])
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
  # Printing progress
  print(paste0("Simulation: ", u, " = Done"))
}

# Summary statistics
fnn_mean2 = mean(RMSE_fnn)
lm_mean2 = mean(RMSE_lm)

fnn_sd2 = sd(RMSE_fnn)/sqrt(250)
lm_sd2 = sd(RMSE_lm)/sqrt(250)

# Times
fnn_time_sim2 = sum(fnn_time)
lm_time_sim2 = sum(lm_time)

# Putting together for boxplot
sim_result2 <- data.frame(FNN = RMSE_fnn, LM = RMSE_lm)

# Boxplot
sqrt_RMSE2 <- data.frame(sim_result2)

# Creating boxplots
plot2 = ggplot(stack(sqrt_RMSE2), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkblue") + 
  theme_bw() + 
  xlab("Model\n (B)") +
  ylab("IMSE") +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

# Running paired t-test

# Selecting data sets
FNN_ttest = RMSE_fnn
FLM_ttest = RMSE_lm

# Calculating difference
d = FNN_ttest - FLM_ttest

# Mean difference
mean_d = mean(d)

# SE
se_d = sd(d)/sqrt(length(FNN_ttest))

# T value
T_value = mean_d/se_d

# df
df_val = length(FNN_ttest) - 1

# p-value
p_value_sim2 = matrix(nrow = 1, ncol = 4)
rownames(p_value_sim2) = c("FNN v. FLM")
colnames(p_value_sim2) = c("P Value", "T Value", "Lower Bound", "Upper Upper Bound")
p_value_sim2[, 1] = pt(abs(T_value), df_val, lower.tail = F)
p_value_sim2[, 2] = T_value
p_value_sim2[, 3] = mean_d - 1.96*se_d
p_value_sim2[, 4] = mean_d + 1.96*se_d

#############################################################
# 3 - Sigmoid 
#############################################################

### Set.Seed
set.seed(1994)
use_session_with_seed(
  1994,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(t) function
beta_coef = runif(5, min = -5, max = 5)

# Generating alpha
alpha = runif(1, 0, 1)

# Initializing
RMSE_fnn <- c()
RMSE_lm <- c()
fnn_time <- c()
lm_time <- c()

# Running loop
for (u in 1:250) {
  
  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*exp(a) + sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = 1/(1 + exp(-(composite_approximator_other(response_func1,
                                                     a = 0,
                                                     b = 1,
                                                     n = 500,
                                                     x_obs = simSmooth$fd$coefs[,i], 
                                                     beta = beta_coef[1]) +
                          composite_approximator_other(response_func2,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[2]) +
                          composite_approximator_other(response_func3,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[3]) +
                          composite_approximator_other(response_func4,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[4]) +
                          composite_approximator_other(response_func5,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[5])) +
                        alpha))
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  
  # Setting up grid
  start_time <- Sys.time()
  l = 2^(-4:10)
  
  # Running functional linear model
  func_basis = fregre.basis.cv(sim_fd, 
                               ystar, 
                               type.basis = "fourier",
                               lambda=l, 
                               type.CV = GCV.S, 
                               par.CV = list(trim=0.15))
  end_time <- Sys.time()
  lm_time[u] = end_time - start_time
  
  # Pulling out the coefficients
  coefficients_lm = func_basis$fregre.basis$coefficients
  
  # Setting up tensor for fnn
  sim_data_fnn = array(dim = c(5, 300, 1))
  sim_data_fnn[,,1] = sim_fd$coefs
  
  # Now running functional neural network
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  start_time <- Sys.time()
  fnn_sim = FNN(resp = ystar, 
                func_cov = sim_data_fnn, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 1,
                neurons_per_layer = c(16),
                activations_in_layers = c("sigmoid"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  end_time <- Sys.time()
  
  # Getting time
  fnn_time[u] = end_time - start_time
  
  # Getting the FNC
  coefficients_fnn = rowMeans(get_weights(fnn_sim$model)[[1]])
  
  # IMSE - FNN
  result_fnn = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_fnn, lower = 0, upper = 1)
  
  # IMSE - LM
  result_lm = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_lm[,1], lower = 0, upper = 1)
  
  #######################################################
  
  # Saving RMSE
  RMSE_fnn[u] <- sqrt(result_fnn[[1]])
  RMSE_lm[u] <- sqrt(result_lm[[1]])
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
  # Printing progress
  print(paste0("Simulation: ", u, " = Done"))
}

# Summary statistics
fnn_mean3 = mean(RMSE_fnn)
lm_mean3 = mean(RMSE_lm)

fnn_sd3 = sd(RMSE_fnn)/sqrt(250)
lm_sd3 = sd(RMSE_lm)/sqrt(250)

# Times
fnn_time_sim3 = sum(fnn_time)
lm_time_sim3 = sum(lm_time)

# Putting together for boxplot
sim_result3 <- data.frame(FNN = RMSE_fnn, LM = RMSE_lm)

# Boxplot
sqrt_RMSE3 <- data.frame(sim_result3)

# Creating boxplots
plot3 = ggplot(stack(sqrt_RMSE3), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkblue") + 
  theme_bw() + 
  xlab("Model\n (C)") +
  ylab("IMSE") +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

# Running paired t-test

# Selecting data sets
FNN_ttest = RMSE_fnn
FLM_ttest = RMSE_lm

# Calculating difference
d = FNN_ttest - FLM_ttest

# Mean difference
mean_d = mean(d)

# SE
se_d = sd(d)/sqrt(length(FNN_ttest))

# T value
T_value = mean_d/se_d

# df
df_val = length(FNN_ttest) - 1

# p-value
p_value_sim3 = matrix(nrow = 1, ncol = 4)
rownames(p_value_sim3) = c("FNN v. FLM")
colnames(p_value_sim3) = c("P Value", "T Value", "Lower Bound", "Upper Upper Bound")
p_value_sim3[, 1] = pt(abs(T_value), df_val, lower.tail = F)
p_value_sim3[, 2] = T_value
p_value_sim3[, 3] = mean_d - 1.96*se_d
p_value_sim3[, 4] = mean_d + 1.96*se_d

#############################################################
# 4 - Log 
#############################################################

### Set.Seed
set.seed(1994)
use_session_with_seed(
  1994,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(t) function
beta_coef = runif(5, min = -5, max = 5)

# Generating alpha
alpha = runif(1, 0, 1)

# Initializing
RMSE_fnn <- c()
RMSE_lm <- c()
fnn_time <- c()
lm_time <- c()

# Running loop
for (u in 1:250) {
  
  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*exp(a) + sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = log(abs(composite_approximator_other(response_func1,
                                                a = 0,
                                                b = 1,
                                                n = 500,
                                                x_obs = simSmooth$fd$coefs[,i], 
                                                beta = beta_coef[1]) +
                     composite_approximator_other(response_func2,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[2]) +
                     composite_approximator_other(response_func3,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[3]) +
                     composite_approximator_other(response_func4,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[4]) +
                     composite_approximator_other(response_func5,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[5])) +
                 alpha)
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  
  # Setting up grid
  start_time <- Sys.time()
  l = 2^(-4:10)
  
  # Running functional linear model
  func_basis = fregre.basis.cv(sim_fd, 
                               ystar, 
                               type.basis = "fourier",
                               lambda=l, 
                               type.CV = GCV.S, 
                               par.CV = list(trim=0.15))
  end_time <- Sys.time()
  lm_time[u] = end_time - start_time
  
  # Pulling out the coefficients
  coefficients_lm = func_basis$fregre.basis$coefficients
  
  # Setting up tensor for fnn
  sim_data_fnn = array(dim = c(5, 300, 1))
  sim_data_fnn[,,1] = sim_fd$coefs
  
  # Now running functional neural network
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  start_time <- Sys.time()
  fnn_sim = FNN(resp = ystar, 
                func_cov = sim_data_fnn, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 3,
                neurons_per_layer = c(16, 16, 16),
                activations_in_layers = c("relu", "linear", "linear"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  end_time <- Sys.time()
  
  # Getting time
  fnn_time[u] = end_time - start_time
  
  # Getting the FNC
  coefficients_fnn = rowMeans(get_weights(fnn_sim$model)[[1]])
  
  # IMSE - FNN
  result_fnn = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_fnn, lower = 0, upper = 1)
  
  # IMSE - LM
  result_lm = integrate(rmse_func, c1 = beta_coef, c2 = coefficients_lm[,1], lower = 0, upper = 1)
  
  #######################################################
  
  # Saving RMSE
  RMSE_fnn[u] <- sqrt(result_fnn[[1]])
  RMSE_lm[u] <- sqrt(result_lm[[1]])
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
  # Printing progress
  print(paste0("Simulation: ", u, " = Done"))
}

# Summary statistics
fnn_mean4 = mean(RMSE_fnn)
lm_mean4 = mean(RMSE_lm)

fnn_sd4 = sd(RMSE_fnn)/sqrt(250)
lm_sd4 = sd(RMSE_lm)/sqrt(250)

# Times
fnn_time_sim4 = sum(fnn_time)
lm_time_sim4 = sum(lm_time)

# Putting together for boxplot
sim_result4 <- data.frame(FNN = RMSE_fnn, LM = RMSE_lm)

# Boxplot
sqrt_RMSE4 <- data.frame(sim_result4)

# Creating boxplots
plot4 = ggplot(stack(sqrt_RMSE4), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkblue") + 
  theme_bw() + 
  xlab("Model\n (D)") +
  ylab("IMSE") +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) # 6 x 4 pdf

# Running paired t-test

# Selecting data sets
FNN_ttest = RMSE_fnn
FLM_ttest = RMSE_lm

# Calculating difference
d = FLM_ttest - FNN_ttest

# Mean difference
mean_d = mean(d)

# SE
se_d = sd(d)/sqrt(length(FNN_ttest))

# T value
T_value = mean_d/se_d

# df
df_val = length(FNN_ttest) - 1

# p-value
p_value_sim4 = matrix(nrow = 1, ncol = 4)
rownames(p_value_sim4) = c("FNN v. FLM")
colnames(p_value_sim4) = c("P Value", "T Value", "Lower Bound", "Upper Upper Bound")
p_value_sim4[, 1] = pt(abs(T_value), df_val, lower.tail = F)
p_value_sim4[, 2] = T_value
p_value_sim4[, 3] = mean_d - 1.96*se_d
p_value_sim4[, 4] = mean_d + 1.96*se_d


# Check 1