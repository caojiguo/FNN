##############################
#                            #
# FNC All Set - Final        #
#                            #
##############################

# Libraries
source("FNN.R")

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(2020)
use_session_with_seed(
  2020,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)


##############################
# Weather Data Set           #
##############################

# Loading data
daily = readRDS("Data/daily.RDS")

# Obtaining response
total_prec = range_01(apply(daily$precav, 2, sum))

# Creating functional data
temp_data = array(dim = c(65, 35, 1))
tempbasis65  = create.fourier.basis(c(0,365), 65)
timepts = seq(1, 365, 1)
temp_fd = Data2fd(timepts, daily$tempav, tempbasis65)

# Changing into fdata
weather_fdata = fdata(daily$tempav, argvals = 1:365, rangeval = c(1, 365))

#######################################

### Functional Linear Model (Basis) ###

# Setting up grid
l=2^(-4:10)

# Running functional linear model
func_basis = fregre.basis.cv(weather_fdata, 
                             total_prec, 
                             type.basis = "fourier",
                             lambda=l, 
                             type.CV = GCV.S, 
                             par.CV = list(trim=0.15))

# Pulling out the coefficients
coefficients_lm = func_basis$fregre.basis$coefficients

# Setting up data set
beta_coef_lm <- data.frame(time = seq(1, 365, 1), 
                           beta_evals = beta_lm_weather(seq(1, 365, 1), c(coefficients_lm[,1])))

#######################################

# Data set up
temp_data[,,1] = temp_fd$coefs

# Running FNN for weather
fnn_weather = FNN(resp = total_prec, 
                  func_cov = temp_data, 
                  scalar_cov = NULL,
                  basis_choice = c("fourier"), 
                  num_basis = 13,
                  hidden_layers = 2,
                  neurons_per_layer = c(16, 8),
                  activations_in_layers = c("relu", "sigmoid"),
                  domain_range = list(c(1, 365)),
                  epochs = 250,
                  output_size = 1,
                  loss_choice = "mse",
                  metric_choice = list("mean_squared_error"),
                  val_split = 0.2,
                  patience_param = 25,
                  learn_rate = 0.05,
                  early_stop = T,
                  print_info = F)

# Getting the FNC
coefficients_fnn = rowMeans(get_weights(fnn_weather$model)[[1]])

# Setting up data set
beta_coef_fnn <- data.frame(time = seq(1, 365, 1), beta_evals = beta_fnn_weather_eg(seq(1, 365, 1), coefficients_fnn))

#### Putting Together #####
beta_coef_fnn %>% 
  ggplot(aes(x = time, y = -beta_evals, color = "blue")) +
  geom_line(size = 1.5) +
  geom_line(data = beta_coef_lm, 
            aes(x = time, y = beta_evals, color = "purple"),
            size = 1.2) + 
  theme_bw() +
  xlab("Time") +
  ylab("beta(t)") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold")) +
  scale_colour_manual(name = 'Model: ', 
                      values =c('purple'='purple','blue'='blue'), 
                      labels = c('Functional Neural Network', 'Functional Linear Model')) +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid", 
                                         colour ="darkblue"),
        legend.position = "bottom",
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12)) # pdf 12 x 4


##############################
# Bike Data Set              #
##############################

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

#######################################

### Functional Linear Model (Basis) ###

# Setting up grid
l=2^(-4:10)

# Running functional linear model
func_basis = fregre.basis.cv(bike_fdata, 
                             rentals, 
                             type.basis = "fourier",
                             lambda=l, 
                             type.CV = GCV.S, 
                             par.CV = list(trim=0.15))

# Pulling out the coefficients
coefficients_lm = func_basis$fregre.basis$coefficients

# Setting up data set
beta_coef_lm <- data.frame(time = seq(1, 24, 0.1), 
                           beta_evals = beta_lm_bike(seq(1, 24, 0.1), c(coefficients_lm[,1])))

#######################################

# Running FNN for bike
# alt: 32, 3
fnn_bike <- FNN(resp = rentals, 
                    func_cov = bike_data, 
                    scalar_cov = NULL,
                    basis_choice = c("fourier"), 
                    num_basis = c(3),
                    hidden_layers = 4,
                    neurons_per_layer = c(32, 32, 32, 32),
                    activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                    domain_range = list(c(1, 24)),
                    epochs = 500,
                    output_size = 1,
                    loss_choice = "mse",
                    metric_choice = list("mean_squared_error"),
                    val_split = 0.15,
                    learn_rate = 0.002,
                    patience_param = 15,
                    early_stop = T,
                    print_info = F)


# Getting the FNC
coefficients_fnn = rowMeans(get_weights(fnn_bike$model)[[1]])

# Setting up data set
beta_coef_fnn <- data.frame(time = seq(1, 24, 0.1), 
                            beta_evals = beta_fnn_bike(seq(1, 24, 0.1), coefficients_fnn))

# Filtering for time
#beta_coef_fnn$beta_evals[which(beta_coef_fnn$time < 6 | beta_coef_fnn$time > 21)] = 0

#### Putting Together #####
beta_coef_fnn %>% 
  ggplot(aes(x = time, y = beta_evals, color='blue')) +
  geom_line(size = 1.5) + 
  theme_bw() +
  xlab("Time") +
  ylab("beta(t)") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold")) +
  scale_colour_manual(name = 'Model: ', 
                      values =c('blue'='blue'), 
                      labels = c('Functional Neural Network')) +
  theme(legend.background = element_rect(fill="lightblue",
                                           size=0.5, linetype="solid", 
                                           colour ="darkblue"),
        legend.position = "bottom",
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12)) # pdf 6 x 4

beta_coef_lm %>% 
  ggplot(aes(x = time, y = beta_evals, color='purple')) +
  geom_line(size = 1.5, color = 'purple') + 
  theme_bw() +
  xlab("Time") +
  ylab("beta(t)") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold")) +
  scale_colour_manual(name = 'Model: ', 
                      values =c('purple'='purple'), 
                      labels = c('Functional Linear Model')) +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid", 
                                         colour ="darkblue"),
        legend.position = "bottom",
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))


# Check 1