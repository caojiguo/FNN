####################################
#                                  #
# Functional Weight Plot - Final   #
#                                  #
####################################

# Libraries
source("FNN.R")

# Loading data
load("Data/bike.RData")

# Obtaining response
rentals = log10(bike$y)

# define the time points on which the functional predictor is observed. 
timepts = bike$timepts

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1,24), nbasis)

# convert the functional predictor into a fda object
bike_fd =  Data2fd(timepts, t(bike$temp), spline_basis)
bike_deriv1 = deriv.fd(bike_fd)
bike_deriv2 = deriv.fd(bike_deriv1)

# Testing with bike data
func_cov_1 = bike_fd$coefs
func_cov_2 = bike_deriv1$coefs
func_cov_3 = bike_deriv2$coefs
#bike_data = array(dim = c(31, 102, 3))
bike_data = array(dim = c(31, 102, 1))
bike_data[,,1] = func_cov_1
#bike_data[,,2] = func_cov_2
#bike_data[,,3] = func_cov_3

# fData Object
bike_fdata = fdata(bike$temp, argvals = 1:24, rangeval = c(1, 24))

# Creating list
plot_list = list()

# vector of epochs
epochs_try = c(1, 25, 50, 99, 120, 140)

for (i in 1:length(epochs_try)) {
  
  # Setting seed
  set.seed(1)
  use_session_with_seed(
    1,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Getting results
  fnn_bike <- FNN(resp = rentals, 
                  func_cov = bike_data, 
                  scalar_cov = NULL,
                  basis_choice = c("fourier"), 
                  num_basis = c(3),
                  hidden_layers = 4,
                  neurons_per_layer = c(1, 32, 32, 32),
                  activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                  domain_range = list(c(1, 24)),
                  epochs = epochs_try[i],
                  output_size = 1,
                  loss_choice = "mse",
                  metric_choice = list("mean_squared_error"),
                  val_split = 0.15,
                  learn_rate = 0.002,
                  patience_param = 15,
                  early_stop = F,
                  print_info = F)
  
  # Getting the FNC
  coefficients_fnn = rowMeans(get_weights(fnn_bike$model)[[1]])
  
  # Setting up data set
  beta_coef_fnn <- data.frame(time = seq(1, 24, 0.1), 
                              beta_evals = beta_fnn_bike(seq(1, 24, 0.1), coefficients_fnn))
  
  # Setting limits
  #beta_coef_fnn$beta_evals[which(beta_coef_fnn$time < 6 | beta_coef_fnn$time > 21)] = 0
  
  #### Putting Together #####
  plot_list[[i]] = beta_coef_fnn %>% 
    ggplot(aes(x = time, y = beta_evals), color='blue') +
    geom_line(size = 1.5, color = "blue") +
    theme_bw() +
    xlab(paste0("Time\n Epochs: ", epochs_try[i])) +
    ylab("beta(t)") +
    ylim(c(-1, 2)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=14, face = "bold"),
          axis.title=element_text(size=14,face="bold"))
  
  # Printing out iteration
  print(paste0("Done Iteration: ", i))
}

ggarrange(plotlist = plot_list, ncol = 6, nrow = 1)

# Finding actual optimal

# Setting seed
set.seed(1)
use_session_with_seed(
  1,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Getting results
fnn_bike <- FNN(resp = rentals, 
                func_cov = bike_data, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = c(3),
                hidden_layers = 4,
                neurons_per_layer = c(1, 32, 32, 32),
                activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                domain_range = list(c(1, 24)),
                epochs = 105,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.15,
                learn_rate = 0.002,
                patience_param = 15,
                early_stop = T,
                print_info = T)

# Check 1