##############################
#                            #
# Ablation Studies - Final   #
#                            #
##############################

##############################
# Data Information:
#
# Bike Data Set
# Observations: 102
# Continuum Points: 24
# Domain: [1, 24]
# Basis Functions used for Functional Observations: 31
# Range of Response: [a, b]
##############################

# Libraries
source("FNN.R")

# Loading data
load("Data/bike.RData")

# Obtaining response
rentals = sqrt(bike$y)

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

# Overall Initialization
ablation_plots <- list()

##############################################################################

### Changing number of basis functions ###

# vector of basis to try
basis_count_try = seq(from = 3, to = 31, by = 2)

# initializing
basis_ablation_df = data.frame(value = NA, mspe = NA)

# looping
for (i in 1:length(basis_count_try)) {
  
  # Setting seed
  set.seed(1)
  use_session_with_seed(
    1,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )

  # Creating folds (50/50 split)
  num_folds = 5
  fold_ind = createFolds(rentals, k = num_folds)
  
  # Initializing
  fnn_mspe = c()
  
  # Going over folds
  for (j in 1:num_folds) {
    
    # Setting up for FNN
    train_x = bike_fdata[-fold_ind[[j]],]
    test_x = bike_fdata[fold_ind[[j]],]
    train_y = rentals[-fold_ind[[j]]]
    test_y = rentals[fold_ind[[j]]]
    bike_data_train = array(dim = c(31, nrow(train_x$data), 1))
    bike_data_test = array(dim = c(31, nrow(test_x$data), 1))
    bike_data_train[,,1] = bike_data[, -fold_ind[[j]], ]
    bike_data_test[,,1] = bike_data[, fold_ind[[j]], ]
    
    # Getting results
    fnn_bike <- FNN(resp = train_y, 
                    func_cov = bike_data_train, 
                    scalar_cov = NULL,
                    basis_choice = c("fourier"), 
                    num_basis = basis_count_try[i],
                    hidden_layers = 4,
                    neurons_per_layer = c(32, 32, 32, 32),
                    activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                    domain_range = list(c(1, 24)),
                    epochs = 250,
                    output_size = 1,
                    loss_choice = "mse",
                    metric_choice = list("mean_squared_error"),
                    val_split = 0.15,
                    learn_rate = 0.002,
                    patience_param = 15,
                    early_stop = T,
                    print_info = F)
    
    # Predicting
    fnn_bike_pred = FNN_Predict(fnn_bike,
                                bike_data_test, 
                                scalar_cov = NULL,
                                basis_choice = c("fourier"), 
                                num_basis = basis_count_try[i],
                                domain_range = list(c(1, 24)))
    
    # MSPE
    fnn_mspe[j] = mean((fnn_bike_pred - test_y)^2, na.rm = T)
    
    # Print iterations done
    print(paste("Fold Iterations Done: ", j))
    
  }
  
  
  
  # Storing
  basis_ablation_df[i, ] = c(basis_count_try[i], mean(fnn_mspe))
  
  # Print iterations done
  print(paste("Overall Iterations Done: ", i))

}

# Plotting
ablation_plots[[1]] = basis_ablation_df %>% 
  ggplot(aes(x = value, y = mspe)) +
  geom_point(size = 1.5, color = "blue") +
  geom_smooth(size = 1.5, color = "blue") +
  theme_bw() +
  xlab("Number of Basis Functions") +
  ylab("MSPE") +
  ggtitle("Functional Weight Basis Count Grid Study") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold"))

### Changing learn rate ###

# vector of learn rates
learn_rate_try = seq(from = 0.0001, to = 1, length.out = 25)

# initializing
learnrate_ablation_df = data.frame(value = NA, mspe = NA)

# looping
for (i in 1:length(learn_rate_try)) {
  
  # Setting seed
  set.seed(1)
  use_session_with_seed(
    1,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Creating folds (50/50 split)
  num_folds = 2
  fold_ind = createFolds(rentals, k = num_folds)
  
  # Initializing
  fnn_mspe = c()
  
  # Going over folds
  for (j in 1:num_folds) {
    
    # Setting up for FNN
    train_x = bike_fdata[-fold_ind[[j]],]
    test_x = bike_fdata[fold_ind[[j]],]
    train_y = rentals[-fold_ind[[j]]]
    test_y = rentals[fold_ind[[j]]]
    bike_data_train = array(dim = c(31, nrow(train_x$data), 1))
    bike_data_test = array(dim = c(31, nrow(test_x$data), 1))
    bike_data_train[,,1] = bike_data[, -fold_ind[[j]], ]
    bike_data_test[,,1] = bike_data[, fold_ind[[j]], ]
    
    # Getting results
    fnn_bike <- FNN(resp = train_y, 
                    func_cov = bike_data_train, 
                    scalar_cov = NULL,
                    basis_choice = c("fourier"), 
                    num_basis = 3,
                    hidden_layers = 4,
                    neurons_per_layer = c(32, 32, 32, 32),
                    activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                    domain_range = list(c(1, 24)),
                    epochs = 250,
                    output_size = 1,
                    loss_choice = "mse",
                    metric_choice = list("mean_squared_error"),
                    val_split = 0.15,
                    learn_rate = learn_rate_try[i],
                    patience_param = 15,
                    early_stop = T,
                    print_info = F)
    
    # Predicting
    fnn_bike_pred = FNN_Predict(fnn_bike,
                                bike_data_test, 
                                scalar_cov = NULL,
                                basis_choice = c("fourier"), 
                                num_basis = 3,
                                domain_range = list(c(1, 24)))
    
    # MSPE
    fnn_mspe[j] = mean((fnn_bike_pred - test_y)^2, na.rm = T)
    
    # Print iterations done
    print(paste("Fold Iterations Done: ", j))
    
  }
  
  
  
  # Storing
  learnrate_ablation_df[i, ] = c(learn_rate_try[i], mean(fnn_mspe))
  
  # Print iterations done
  print(paste("Overall Iterations Done: ", i))
  
}

# Plotting
ablation_plots[[2]] = learnrate_ablation_df %>% 
  ggplot(aes(x = value, y = mspe)) +
  geom_point(size = 1.5, color = "red") +
  geom_smooth(size = 1.5, color = "red") +
  theme_bw() +
  xlab("Learn Rate") +
  ylab("MSPE") +
  ggtitle("Learn Rate Grid Study") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold"))

### Changing val_split ###

# vector of validation splits
val_split_try = seq(from = 0.05, to = 0.5, length.out = 10)

# initializing
valsplit_ablation_df = data.frame(value = NA, mspe = NA)

# looping
for (i in 1:length(val_split_try)) {
  
  # Setting seed
  set.seed(1)
  use_session_with_seed(
    1,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Creating folds (50/50 split)
  num_folds = 2
  fold_ind = createFolds(rentals, k = num_folds)
  
  # Initializing
  fnn_mspe = c()
  
  # Going over folds
  for (j in 1:num_folds) {
    
    # Setting up for FNN
    train_x = bike_fdata[-fold_ind[[j]],]
    test_x = bike_fdata[fold_ind[[j]],]
    train_y = rentals[-fold_ind[[j]]]
    test_y = rentals[fold_ind[[j]]]
    bike_data_train = array(dim = c(31, nrow(train_x$data), 1))
    bike_data_test = array(dim = c(31, nrow(test_x$data), 1))
    bike_data_train[,,1] = bike_data[, -fold_ind[[j]], ]
    bike_data_test[,,1] = bike_data[, fold_ind[[j]], ]
   
    # Getting results
    fnn_bike <- FNN(resp = train_y, 
                    func_cov = bike_data_train, 
                    scalar_cov = NULL,
                    basis_choice = c("fourier"), 
                    num_basis = 3,
                    hidden_layers = 4,
                    neurons_per_layer = c(32, 32, 32, 32),
                    activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                    domain_range = list(c(1, 24)),
                    epochs = 250,
                    output_size = 1,
                    loss_choice = "mse",
                    metric_choice = list("mean_squared_error"),
                    val_split = val_split_try[i],
                    learn_rate = 0.15,
                    patience_param = 15,
                    early_stop = T,
                    print_info = F)
    
    # Predicting
    fnn_bike_pred = FNN_Predict(fnn_bike,
                                bike_data_test, 
                                scalar_cov = NULL,
                                basis_choice = c("fourier"), 
                                num_basis = 3,
                                domain_range = list(c(1, 24)))
    
    # MSPE
    fnn_mspe[j] = mean((fnn_bike_pred - test_y)^2, na.rm = T)
    
    # Print iterations done
    print(paste("Fold Iterations Done: ", j))
    
  }
  
  
  
  # Storing
  valsplit_ablation_df[i, ] = c(val_split_try[i], mean(fnn_mspe))
  
  # Print iterations done
  print(paste("Overall Iterations Done: ", i))
  
}

# Plotting
ablation_plots[[3]] = valsplit_ablation_df %>% 
  ggplot(aes(x = value, y = mspe)) +
  geom_point(size = 1.5, color = "black") +
  geom_smooth(size = 1.5, color = "black") +
  theme_bw() +
  xlab("Validation Split") +
  ylab("MSPE") +
  ggtitle("Validation Split Grid Study") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold"))

### Changing number of epochs ###

# vector of epochs
epochs_try = seq(from = 5, to = 500, length.out = 10)

# initializing
epochs_ablation_df = data.frame(value = NA, mspe = NA)

# looping
for (i in 1:length(epochs_try)) {
  
  # Setting seed
  set.seed(1)
  use_session_with_seed(
    1,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Creating folds (50/50 split)
  num_folds = 2
  fold_ind = createFolds(rentals, k = num_folds)
  
  # Initializing
  fnn_mspe = c()
  
  # Going over folds
  for (j in 1:num_folds) {
    
    # Setting up for FNN
    train_x = bike_fdata[-fold_ind[[j]],]
    test_x = bike_fdata[fold_ind[[j]],]
    train_y = rentals[-fold_ind[[j]]]
    test_y = rentals[fold_ind[[j]]]
    bike_data_train = array(dim = c(31, nrow(train_x$data), 1))
    bike_data_test = array(dim = c(31, nrow(test_x$data), 1))
    bike_data_train[,,1] = bike_data[, -fold_ind[[j]], ]
    bike_data_test[,,1] = bike_data[, fold_ind[[j]], ]
    
    # Getting results
    fnn_bike <- FNN(resp = train_y, 
                    func_cov = bike_data_train, 
                    scalar_cov = NULL,
                    basis_choice = c("fourier"), 
                    num_basis = 3,
                    hidden_layers = 4,
                    neurons_per_layer = c(32, 32, 32, 32),
                    activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                    domain_range = list(c(1, 24)),
                    epochs = epochs_try[i],
                    output_size = 1,
                    loss_choice = "mse",
                    metric_choice = list("mean_squared_error"),
                    val_split = 0.15,
                    learn_rate = 0.0005,
                    patience_param = 15,
                    early_stop = F,
                    print_info = F)
    
    # Predicting
    fnn_bike_pred = FNN_Predict(fnn_bike,
                                bike_data_test, 
                                scalar_cov = NULL,
                                basis_choice = c("fourier"), 
                                num_basis = 3,
                                domain_range = list(c(1, 24)))
    
    # MSPE
    fnn_mspe[j] = mean((fnn_bike_pred - test_y)^2, na.rm = T)
    
    # Print iterations done
    print(paste("Fold Iterations Done: ", j))
    
  }
  
  
  
  # Storing
  epochs_ablation_df[i, ] = c(epochs_try[i], mean(fnn_mspe))
  
  # Print iterations done
  print(paste("Overall Iterations Done: ", i))
  
}

# Plotting
ablation_plots[[4]] = epochs_ablation_df %>% 
  ggplot(aes(x = value, y = mspe)) +
  geom_point(size = 1.5, color = "purple") +
  geom_smooth(size = 1.5, color = "purple") +
  theme_bw() +
  xlab("Epochs") +
  ylab("MSPE") +
  ggtitle("Epochs Grid Study") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold"))

### Changing neurons ###

# vector of neurons
neurons_try = seq(from = 2, to = 256, length.out = 10)

# initializing
neurons_ablation_df = data.frame(value = NA, mspe = NA)

# looping
for (i in 1:length(neurons_try)) {
  
  # Setting seed
  set.seed(1)
  use_session_with_seed(
    1,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Creating folds (50/50 split)
  num_folds = 2
  fold_ind = createFolds(rentals, k = num_folds)
  
  # Initializing
  fnn_mspe = c()
  
  # Going over folds
  for (j in 1:num_folds) {
    
    # Setting up for FNN
    train_x = bike_fdata[-fold_ind[[j]],]
    test_x = bike_fdata[fold_ind[[j]],]
    train_y = rentals[-fold_ind[[j]]]
    test_y = rentals[fold_ind[[j]]]
    bike_data_train = array(dim = c(31, nrow(train_x$data), 1))
    bike_data_test = array(dim = c(31, nrow(test_x$data), 1))
    bike_data_train[,,1] = bike_data[, -fold_ind[[j]], ]
    bike_data_test[,,1] = bike_data[, fold_ind[[j]], ]
    
    # Getting results
    fnn_bike <- FNN(resp = train_y, 
                    func_cov = bike_data_train, 
                    scalar_cov = NULL,
                    basis_choice = c("fourier"), 
                    num_basis = 3,
                    hidden_layers = 4,
                    neurons_per_layer = c(neurons_try[i], neurons_try[i], neurons_try[i], neurons_try[i]),
                    activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                    domain_range = list(c(1, 24)),
                    epochs = 250,
                    output_size = 1,
                    loss_choice = "mse",
                    metric_choice = list("mean_squared_error"),
                    val_split = 0.15,
                    learn_rate = 0.15,
                    patience_param = 15,
                    early_stop = T,
                    print_info = F)
    
    # Predicting
    fnn_bike_pred = FNN_Predict(fnn_bike,
                                bike_data_test, 
                                scalar_cov = NULL,
                                basis_choice = c("fourier"), 
                                num_basis = 3,
                                domain_range = list(c(1, 24)))
    
    # MSPE
    fnn_mspe[j] = mean((fnn_bike_pred - test_y)^2, na.rm = T)
    
    # Print iterations done
    print(paste("Fold Iterations Done: ", j))
    
  }
  
  
  
  # Storing
  neurons_ablation_df[i, ] = c(neurons_try[i], mean(fnn_mspe))
  
  # Print iterations done
  print(paste("Overall Iterations Done: ", i))
  
}

# Plotting
ablation_plots[[5]] = neurons_ablation_df %>% 
  ggplot(aes(x = value, y = mspe)) +
  geom_point(size = 1.5, color = "orange") +
  geom_smooth(size = 1.5, color = "orange") +
  theme_bw() +
  xlab("Neurons") +
  ylab("MSPE") +
  ggtitle("Neuron Count Grid Study") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold"))

### Changing decay rate ###

# vector of epochs
decay_rate_try = seq(from = 0, to = 1, length.out = 20)

# initializing
decayrate_ablation_df = data.frame(value = NA, mspe = NA)

# looping
for (i in 1:length(decay_rate_try)) {
  
  # Setting seed
  set.seed(1)
  use_session_with_seed(
    1,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Creating folds (50/50 split)
  num_folds = 2
  fold_ind = createFolds(rentals, k = num_folds)
  
  # Initializing
  fnn_mspe = c()
  
  # Going over folds
  for (j in 1:num_folds) {
    
    # Setting up for FNN
    train_x = bike_fdata[-fold_ind[[j]],]
    test_x = bike_fdata[fold_ind[[j]],]
    train_y = rentals[-fold_ind[[j]]]
    test_y = rentals[fold_ind[[j]]]
    bike_data_train = array(dim = c(31, nrow(train_x$data), 1))
    bike_data_test = array(dim = c(31, nrow(test_x$data), 1))
    bike_data_train[,,1] = bike_data[, -fold_ind[[j]], ]
    bike_data_test[,,1] = bike_data[, fold_ind[[j]], ]
    
    # Getting results
    fnn_bike <- FNN(resp = train_y, 
                    func_cov = bike_data_train, 
                    scalar_cov = NULL,
                    basis_choice = c("fourier"), 
                    num_basis = 3,
                    hidden_layers = 4,
                    neurons_per_layer = c(32, 32, 32, 32),
                    activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                    domain_range = list(c(1, 24)),
                    epochs = 250,
                    output_size = 1,
                    loss_choice = "mse",
                    metric_choice = list("mean_squared_error"),
                    val_split = 0.15,
                    learn_rate = 0.15,
                    patience_param = 15,
                    early_stop = T,
                    print_info = F,
                    decay_rate = decay_rate_try[i])
    
    # Predicting
    fnn_bike_pred = FNN_Predict(fnn_bike,
                                bike_data_test, 
                                scalar_cov = NULL,
                                basis_choice = c("fourier"), 
                                num_basis = 3,
                                domain_range = list(c(1, 24)))
    
    # MSPE
    fnn_mspe[j] = mean((fnn_bike_pred - test_y)^2, na.rm = T)
    
    # Print iterations done
    print(paste("Fold Iterations Done: ", j))
    
  }
  

  
  # Storing
  decayrate_ablation_df[i, ] = c(decay_rate_try[i], mean(fnn_mspe))
  
  # Print iterations done
  print(paste("Overall Iterations Done: ", i))
  
}

# Plotting
ablation_plots[[6]] = decayrate_ablation_df %>% 
  ggplot(aes(x = value, y = mspe)) +
  geom_point(size = 1.5, color = "pink") +
  geom_smooth(size = 1.5, color = "pink") +
  theme_bw() +
  xlab("Decay Rate") +
  ylab("MSPE") +
  ggtitle("Decay Rate Grid Study") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold"))


# Final Plot
n_plots <- length(ablation_plots)
nCol <- 2
do.call("grid.arrange", c(ablation_plots, ncol = nCol)) # pdf 12 x 13

# Check 1