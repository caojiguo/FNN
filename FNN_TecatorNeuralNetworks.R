###################################
#                                 #
# Tecator Data Set (NN) - Final   #
#                                 #
###################################

##############################
# Some Data Information:
#
# Tecator Data Set
# Observations: 215
# Continuum Points: 100
# Continuum Domain: [850, 1050]
# Basis Functions used for Functional Observations: 29, Fourier
# Range of Response: [0.9, 49.1]
# Basis Functions used for Functional Weights: 3
# Folds Used: Compared with another paper (only test/train split)
# Parameter Count in NN: 4029
# Parameter Count in CNN: 87045
# Parameter Count in FNN: 5757
# Parameter Count in LSTM: 5757
# Parameter Count in LSTM Bidirectional: 5757
# Parameter Count in GRU: 5757
##############################

# Source for FNN
source("FNN.R")

# Loading data
tecator = readRDS("Data/tecator.RDS")

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(17)
use_session_with_seed(
  17,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# define the time points on which the functional predictor is observed. 
timepts = tecator$absorp.fdata$argvals

# define the fourier basis 
nbasis = 29
spline_basis = create.fourier.basis(tecator$absorp.fdata$rangeval, nbasis)

# convert the functional predictor into a fda object and getting deriv
tecator_fd =  Data2fd(timepts, t(tecator$absorp.fdata$data), spline_basis)
tecator_deriv = deriv.fd(tecator_fd)
tecator_deriv2 = deriv.fd(tecator_deriv)

# Non functional covariate
tecator_scalar = data.frame(water = tecator$y$Water)

# Response
tecator_resp = tecator$y$Fat

# Scaling data
mean = apply(tecator$absorp.fdata$data, 2, mean)
std = apply(tecator$absorp.fdata$data, 2, sd)
tecator_scaled = as.matrix(scale(tecator$absorp.fdata$data, center = mean, scale = std))

# Non functional covariate
tecator_scalar = data.frame(water = tecator$y$Water)

# Scaling this variable
mean_water = mean(tecator_scalar$water)
std_water = sd(tecator_scalar$water)
water_scaled = c(scale(tecator_scalar$water, center = mean_water, scale = std_water))

# Getting data into right format
tecator_data = array(dim = c(nbasis, 215, 1))
tecator_data[,,1] = tecator_fd$coefs

# Choosing fold number
num_folds = 5

# Creating folds
fold_ind = createFolds(tecator_resp, k = num_folds)

# Initializing matrices for results
error_mat_fnn = matrix(nrow = num_folds, ncol = 2)
error_mat_nn = matrix(nrow = num_folds, ncol = 2)
error_mat_cnn = matrix(nrow = num_folds, ncol = 2)
error_mat_cnn_2f_32 = matrix(nrow = num_folds, ncol = 2)
error_mat_cnn_3f = matrix(nrow = num_folds, ncol = 2) # size 1 kernel
error_mat_cnn_5f = matrix(nrow = num_folds, ncol = 2) # size 2 kernel, 16 filters
error_mat_rnn_lstm = matrix(nrow = num_folds, ncol = 2)
error_mat_rnn_lstm_bidirectional = matrix(nrow = num_folds, ncol = 2)
error_mat_rnn_gru = matrix(nrow = num_folds, ncol = 2)

# Functional weights & initializations
fnn_training_plot <- list()
nn_training_plot <- list()
rnn_lstm_training_plot <- list()
rnn_bidirectional_training_plot <- list()
rnn_gru_training_plot <- list()
cnn_training_plot <- list()

# For testing
# i = 1
# u = 1

# Looping to get results
for (i in 1:num_folds) {
  
  ################## 
  # Splitting data #
  ##################
  
  tec_data_train_fnn <- array(dim = c(nbasis, nrow(tecator$absorp.fdata$data) - length(fold_ind[[i]]), 1))
  tec_data_test_fnn <- array(dim = c(nbasis, length(fold_ind[[i]]), 1))
  tec_data_train_fnn[,,1] = tecator_data[, -fold_ind[[i]], ]
  tec_data_test_fnn[,,1] = tecator_data[, fold_ind[[i]], ]
  train_y = tecator_resp[-fold_ind[[i]]]
  test_y = tecator_resp[fold_ind[[i]]]
  scalar_train = data.frame(tecator_scalar[-fold_ind[[i]],1])
  scalar_test = data.frame(tecator_scalar[fold_ind[[i]],1])
  
  ########################################
  # Neural Network Tuning Setup          #
  ########################################
  
  # Initializing
  min_error_nn = 99999
  min_error_cnn = 99999
  min_error_cnn_2f_32 = 99999
  min_error_cnn_3f = 99999
  min_error_cnn_5f = 99999
  min_error_lstm = 99999
  min_error_lstm_bidirectional = 99999
  min_error_gru = 99999
  
  # Setting up MV data
  # MV_train = as.data.frame(cbind(as.data.frame((tecator$absorp.fdata$data)[-fold_ind[[i]],]), water = scalar_train))
  # MV_test = as.data.frame(cbind(as.data.frame((tecator$absorp.fdata$data)[fold_ind[[i]],]), water = scalar_test))
  
  # Setting up MV data
  MV_train = as.data.frame(cbind(as.data.frame((tecator_scaled)[-fold_ind[[i]],]), water = water_scaled[-fold_ind[[i]]]))
  MV_test = as.data.frame(cbind(as.data.frame((tecator_scaled)[fold_ind[[i]],]), water = water_scaled[fold_ind[[i]]]))
  
  # Setting up for RNN
  tec_data_train <- array(dim = c(nrow(tecator$absorp.fdata$data) - length(-fold_ind[[i]]), ncol(MV_train), 1))
  tec_data_test <- array(dim =c(length(-fold_ind[[i]]), ncol(MV_train), 1))
  tec_data_train[,,1] = data.matrix(MV_train)
  tec_data_test[,,1] = data.matrix(MV_test)
  
  # Responses
  tecResp_train = tecator_resp[-fold_ind[[i]]]
  tecResp_test = tecator_resp[fold_ind[[i]]]
  
  # Random Split
  train_split = sample(1:nrow(MV_train), floor(0.5*nrow(MV_train)))
  
  # Initialization Count
  num_initalizations = 10
  
  ########################################
  # Running Regular Neural Network       #
  ########################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_nn <- keras_model_sequential()
    model_nn %>% 
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 58, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_nn %>% compile(
      optimizer = optimizer_adam(lr = 0.005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Training FNN model
    history_nn <- model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                                   tecResp_train[train_split], 
                                   epochs = 300,  
                                   validation_split = 0.2,
                                   callbacks = list(early_stop),
                                   verbose = 0)
    
    # Predictions
    test_predictions <- model_nn %>% predict(as.matrix(MV_train[-train_split,]))
    
    # Plotting
    error_nn_train = mean((c(test_predictions) - tecResp_train[-train_split])^2)
    
    # Checking error
    if(error_nn_train < min_error_nn){
      
      # Predictions
      pred_nn <- model_nn %>% predict(as.matrix(MV_test))
      
      # Error
      MSPE_NN = mean((c(pred_nn) - tecResp_test)^2, na.rm = T)
      Rsquared_NN = 1 - sum((pred_nn - tecResp_test)^2)/sum((tecResp_test - mean(tecResp_test))^2)
      
      # Saving training plots
      nn_training_plot[[i]] = as.data.frame(history_nn)
      
      # New Min Error
      min_error_nn = error_nn_train
      
    }
    
  }
  
  ########################################
  # Running Convolutional Neural Network #
  ########################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_cnn <- keras_model_sequential()
    model_cnn %>% 
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 58, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn %>% compile(
      optimizer = optimizer_adam(lr = 0.005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn <- model_cnn %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 300,  
                                     validation_split = 0.2,
                                     callbacks = list(early_stop),
                                     verbose = 0)
    
    # Predictions
    test_predictions <- model_cnn %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_cnn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_cnn_train < min_error_cnn){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(MV_test), ncol(MV_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(MV_test)
      
      # Predictions
      pred_cnn <- model_cnn %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn = mean((c(pred_cnn) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn = error_cnn_train
      
    }
    
  }
  
  
  #######################################################
  # Running Convolutional Neural Network 2f 32 filters #
  #######################################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_cnn_2f_32 <- keras_model_sequential()
    model_cnn_2f_32 %>% 
      layer_conv_1d(filters = 32, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 32, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 58, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn_2f_32 %>% compile(
      optimizer = optimizer_adam(lr = 0.005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn_2f_32 <- model_cnn_2f_32 %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 300,  
                                     validation_split = 0.2,
                                     callbacks = list(early_stop),
                                     verbose = 0)
    
    # Predictions
    test_predictions <- model_cnn_2f_32 %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_cnn_train_2f_32 = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_cnn_train_2f_32 < min_error_cnn_2f_32){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(MV_test), ncol(MV_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(MV_test)
      
      # Predictions
      pred_cnn_2f_32 <- model_cnn_2f_32 %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      # cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn_2f_32 = mean((c(pred_cnn_2f_32) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn_2f_32 = error_cnn_train_2f_32
      
    }
    
  }
  
  ############################################
  # Running Convolutional Neural Network 3f #
  ############################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_cnn_3f <- keras_model_sequential()
    model_cnn_3f %>% 
      layer_conv_1d(filters = 64, kernel_size = 3, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 64, kernel_size = 3, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 58, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn_3f %>% compile(
      optimizer = optimizer_adam(lr = 0.005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn_3f <- model_cnn_3f %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 300,  
                                     validation_split = 0.2,
                                     callbacks = list(early_stop),
                                     verbose = 0)
    
    # Predictions
    test_predictions_3f <- model_cnn_3f %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_cnn_train_3f = mean((c(test_predictions_3f) - train_y[-train_split])^2)
    
    # Checking error
    if(error_cnn_train_3f < min_error_cnn_3f){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(MV_test), ncol(MV_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(MV_test)
      
      # Predictions
      pred_cnn_3f <- model_cnn_3f %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      # cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn_3f = mean((c(pred_cnn_3f) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn_3f = error_cnn_train_3f
      
    }
    
  }
  
  ############################################
  # Running Convolutional Neural Network 5f #
  ############################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_cnn_5f <- keras_model_sequential()
    model_cnn_5f %>% 
      layer_conv_1d(filters = 16, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 16, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 24, activation = 'relu') %>%
      layer_dense(units = 58, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn_5f %>% compile(
      optimizer = optimizer_adam(lr = 0.005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn_5f <- model_cnn_5f %>% fit(reshaped_data_tensor_train, 
                                           train_y[train_split], 
                                           epochs = 300,  
                                           validation_split = 0.2,
                                           callbacks = list(early_stop),
                                           verbose = 0)
    
    # Predictions
    test_predictions_5f <- model_cnn_5f %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_cnn_train_5f = mean((c(test_predictions_5f) - train_y[-train_split])^2)
    
    # Checking error
    if(error_cnn_train_5f < min_error_cnn_5f){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(MV_test), ncol(MV_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(MV_test)
      
      # Predictions
      pred_cnn_5f <- model_cnn_5f %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      # cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn_5f = mean((c(pred_cnn_5f) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn_5f = error_cnn_train_5f
      
    }
    
  }
  
  
  ########################################
  # Running Simple LSTM Neural Network   #
  ########################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # Regular Model
    model_lstm <- keras_model_sequential() 
    model_lstm %>% 
      layer_lstm(units = 64, input_shape = c(101, 1)) %>% 
      layer_dense(units = 64, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 8, activation = "relu") %>%
      layer_dense(units = 1)
    
    # Compile
    model_lstm %>% compile(
      optimizer = optimizer_adam(lr = 0.01), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(tec_data_train[train_split,,]), ncol(tec_data_train[train_split,,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(tec_data_train[train_split,,])
    reshaped_data_tensor_test = array(dim = c(nrow(tec_data_train[-train_split,,]), ncol(tec_data_train[-train_split,,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(tec_data_train[-train_split,,])
    
    # Fit model
    history_lstm = model_lstm %>% fit(reshaped_data_tensor_train, 
                                 tecResp_train[train_split],
                                 epochs = 300,
                                 batch_size = 32,
                                 validation_split = 0.2,
                                 callbacks = list(early_stop),
                                 verbose = 0)
    
    # Predictions
    test_predictions <- model_lstm %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_lstm_train = mean((c(test_predictions) - tecResp_train[-train_split])^2)
    
    # Checking error
    if(error_lstm_train < min_error_lstm){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(tec_data_test), ncol(tec_data_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(tec_data_test)
      
      # Predictions
      pred_lstm <- model_lstm %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      rnn_lstm_training_plot[[i]] = as.data.frame(history_lstm)
      
      # Error
      error_lstm = mean((c(pred_lstm) - tecResp_test)^2, na.rm = T)
      
      # New Min Error
      min_error_lstm = error_lstm_train
      
    }
    
  }
  
  
  ###############################################
  # Running Bidirectional LSTM Neural Network   #
  ###############################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # Regular Model
    model_lstm_bidirectional <- keras_model_sequential() 
    model_lstm_bidirectional %>% 
      bidirectional(layer_lstm(units = 64, input_shape = c(101, 1))) %>% 
      layer_dense(units = 64, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 8, activation = "relu") %>%
      layer_dense(units = 1)
    
    # Compile
    model_lstm_bidirectional %>% compile(
      optimizer = optimizer_adam(lr = 0.01), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(tec_data_train[train_split,,]), ncol(tec_data_train[train_split,,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(tec_data_train[train_split,,])
    reshaped_data_tensor_test = array(dim = c(nrow(tec_data_train[-train_split,,]), ncol(tec_data_train[-train_split,,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(tec_data_train[-train_split,,])
    
    # Fit model
    history_lstm_bidirectional = model_lstm_bidirectional %>% fit(reshaped_data_tensor_train, 
                                      tecResp_train[train_split],
                                      epochs = 300,
                                      batch_size = 32,
                                      validation_split = 0.2,
                                      callbacks = list(early_stop),
                                      verbose = 0)
    
    # Predictions
    test_predictions <- model_lstm_bidirectional %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_lstm_bidirectional_train = mean((c(test_predictions) - tecResp_train[-train_split])^2)
    
    # Checking error
    if(error_lstm_bidirectional_train < min_error_lstm_bidirectional){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(tec_data_test), ncol(tec_data_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(tec_data_test)
      
      # Predictions
      pred_lstm_bidirectional <- model_lstm_bidirectional %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      rnn_bidirectional_training_plot[[i]] = as.data.frame(history_lstm_bidirectional)
      
      # Error
      error_lstm_bidirectional = mean((c(pred_lstm_bidirectional) - tecResp_test)^2, na.rm = T)
      
      # New Min Error
      min_error_lstm_bidirectional = error_lstm_bidirectional
      
    }
    
  }
  
  ###############################################
  # Running Gated Recurrent Unit                #
  ###############################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # Regular Model
    model_gru <- keras_model_sequential() 
    model_gru %>% 
      layer_gru(units = 64, input_shape = c(101, 1)) %>%
      layer_dense(units = 64, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 8, activation = "relu") %>%
      layer_dense(units = 1)
    
    # Compile
    model_gru %>% compile(
      optimizer = optimizer_adam(lr = 0.01), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(tec_data_train[train_split,,]), ncol(tec_data_train[train_split,,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(tec_data_train[train_split,,])
    reshaped_data_tensor_test = array(dim = c(nrow(tec_data_train[-train_split,,]), ncol(tec_data_train[-train_split,,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(tec_data_train[-train_split,,])
    
    # Fit model
    history_gru = model_gru %>% fit(reshaped_data_tensor_train, 
                                    tecResp_train[train_split],
                                    epochs = 300,
                                    batch_size = 32,
                                    validation_split = 0.2,
                                    callbacks = list(early_stop),
                                    verbose = 0)
    
    # Predictions
    test_predictions <- model_gru %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_gru_train = mean((c(test_predictions) - tecResp_train[-train_split])^2)
    
    # Checking error
    if(error_gru_train < min_error_gru){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(tec_data_test), ncol(tec_data_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(tec_data_test)
      
      # Predictions
      pred_gru <- model_gru %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      rnn_gru_training_plot[[i]] = as.data.frame(history_gru)
      
      # Error
      error_gru = mean((c(pred_gru) - tecResp_test)^2, na.rm = T)
      
      # New Min Error
      min_error_gru = error_gru
      
    }
    
  }

  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )

  # FNN for tecator data
  tecator_comp = FNN(resp = train_y, 
                     func_cov = tec_data_train_fnn, 
                     scalar_cov = scalar_train,
                     basis_choice = c("fourier"), 
                     num_basis = 3,
                     hidden_layers = 6,
                     neurons_per_layer = c(24, 24, 24, 24, 24, 58),
                     activations_in_layers = c("relu", "relu", "relu", "relu", "relu", "linear"),
                     domain_range = list(c(850, 1050)),
                     epochs = 300,
                     output_size = 1,
                     loss_choice = "mse",
                     metric_choice = list("mean_squared_error"),
                     val_split = 0.2,
                     patience_param = 10,
                     learn_rate = 0.005,
                     decay_rate = 0,
                     batch_size = 32,
                     early_stop = T,
                     print_info = F)
  
  # Predicting
  pred_tec = FNN_Predict(tecator_comp,
                         tec_data_test_fnn, 
                         scalar_cov = scalar_test,
                         basis_choice = c("fourier"), 
                         num_basis = 3,
                         domain_range = list(c(850, 1050)))
  
  # Training plots
  fnn_training_plot[[i]] = data.frame(epoch = 1:300, value = c(tecator_comp$per_iter_info$val_loss, rep(NA, 300 - length(tecator_comp$per_iter_info$val_loss))))
  
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_fnn[i, 1] = mean((c(pred_tec) - test_y)^2, na.rm = T)
  error_mat_nn[i, 1] = mean((pred_nn - test_y)^2, na.rm = T)
  error_mat_cnn[i, 1] = mean((pred_cnn - test_y)^2, na.rm = T)
  error_mat_cnn_2f_32[i, 1] = mean((pred_cnn_2f_32 - test_y)^2, na.rm = T)
  error_mat_cnn_3f[i, 1] = mean((pred_cnn_3f - test_y)^2, na.rm = T)
  error_mat_cnn_5f[i, 1] = mean((pred_cnn_5f - test_y)^2, na.rm = T)
  error_mat_rnn_lstm[i, 1] = mean((pred_lstm - test_y)^2, na.rm = T)
  error_mat_rnn_lstm_bidirectional[i, 1] = mean((pred_lstm_bidirectional - test_y)^2, na.rm = T)
  error_mat_rnn_gru[i, 1] = mean((pred_gru - test_y)^2, na.rm = T)
  
  # R^2 Results
  error_mat_fnn[i, 2] = 1 - sum((c(pred_tec) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_nn[i, 2] = 1 - sum((pred_nn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_cnn[i, 2] = 1 - sum((pred_cnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_cnn_2f_32[i, 2] = 1 - sum((pred_cnn_2f_32 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_cnn_3f[i, 2] = 1 - sum((pred_cnn_3f - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_cnn_5f[i, 2] = 1 - sum((pred_cnn_5f - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_rnn_lstm[i, 2] = 1 - sum((pred_lstm - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_rnn_lstm_bidirectional[i, 2] = 1 - sum((pred_lstm_bidirectional - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_rnn_gru[i, 2] = 1 - sum((pred_gru - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", i))
  
  # Clearning sessions
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Tecator = matrix(nrow = 9, ncol = 3)

# Collecting errors, R^2, and SE
Final_Table_Tecator[1, ] = c(colMeans(error_mat_fnn, na.rm = T), sd(error_mat_fnn[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[2, ] = c(colMeans(error_mat_nn, na.rm = T), sd(error_mat_nn[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[3, ] = c(colMeans(error_mat_cnn, na.rm = T), sd(error_mat_cnn[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[4, ] = c(colMeans(error_mat_cnn_2f_32, na.rm = T), sd(error_mat_cnn_2f_32[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[5, ] = c(colMeans(error_mat_cnn_3f, na.rm = T), sd(error_mat_cnn_3f[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[6, ] = c(colMeans(error_mat_cnn_5f, na.rm = T), sd(error_mat_cnn_5f[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[7, ] = c(colMeans(error_mat_rnn_lstm, na.rm = T), sd(error_mat_rnn_lstm[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[8, ] = c(colMeans(error_mat_rnn_lstm_bidirectional, na.rm = T), sd(error_mat_rnn_lstm_bidirectional[,1], na.rm = T)/sqrt(num_folds))
Final_Table_Tecator[9, ] = c(colMeans(error_mat_rnn_gru, na.rm = T), sd(error_mat_rnn_gru[,1], na.rm = T)/sqrt(num_folds))

# Looking at results
colnames(Final_Table_Tecator) <- c("CV_MSPE", "R2", "SE")
rownames(Final_Table_Tecator) <- c("FNN", "NN", "CNN", "CNN; Kernel Size: 2, Filters: 32", "CNN; Kernel Size: 3", "CNN; Kernel Size: 2, Filters: 16",
                                   "LSTM", "BIDIRECTIONAL LSTM", "GRU")
Final_Table_Tecator

### Training plots saving

# Initializing plots
training_plots_tecator = list()

# Looping
for (i in 1:num_folds) {
  
  # count
  a = 6*(i - 1)
  
  # Saving relevant
  current_nn = nn_training_plot[[i]]
  current_cnn = cnn_training_plot[[i]]
  current_fnn = fnn_training_plot[[i]]
  current_lstm = rnn_lstm_training_plot[[i]]
  current_lstm_bidirectional = rnn_bidirectional_training_plot[[i]]
  current_gru = rnn_gru_training_plot[[i]]
  
  # Filtering
  current_nn = current_nn %>% dplyr::filter(metric == "loss" & data == "validation")
  current_cnn = current_cnn %>% dplyr::filter(metric == "loss" & data == "validation")
  current_fnn = current_fnn
  current_lstm = current_lstm %>% dplyr::filter(metric == "loss" & data == "validation")
  current_lstm_bidirectional = current_lstm_bidirectional %>% dplyr::filter(metric == "loss" & data == "validation")
  current_gru = current_gru %>% dplyr::filter(metric == "loss" & data == "validation")
  
  # Creating plots
  nn_plot = current_nn %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5,  color='red') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("Neural Network;\n Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=8, face = "bold"),
          axis.title=element_text(size=12,face="bold")) +
    theme(plot.title = element_text(size=8))
  
  cnn_plot = current_cnn %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5, color='green') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("Convolutional Neural Network;\n Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=8, face = "bold"),
          axis.title=element_text(size=12,face="bold")) +
    theme(plot.title = element_text(size=8))
  
  fnn_plot = current_fnn %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5, color='blue') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("Functional Neural Network;\n Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=8, face = "bold"),
          axis.title=element_text(size=12,face="bold")) +
    theme(plot.title = element_text(size=8))
  
  lstm_plot = current_lstm %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5,  color='purple') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("LSTM Neural Network;\n Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=8, face = "bold"),
          axis.title=element_text(size=12,face="bold")) +
    theme(plot.title = element_text(size=8))
  
  lstm_bidirectional_plot = current_lstm_bidirectional %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5, color='pink') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("LSTM Bidirectional Neural Network;\n Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=8, face = "bold"),
          axis.title=element_text(size=12,face="bold")) +
    theme(plot.title = element_text(size=8))
  
  gru_plot = current_gru %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5, color='black') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("Gated Recurrent Unit;\n Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=8, face = "bold"),
          axis.title=element_text(size=12,face="bold")) +
    theme(plot.title = element_text(size=8))
  
  
  # Storing
  training_plots_tecator[[a + 1]] = nn_plot
  training_plots_tecator[[a + 2]] = cnn_plot
  training_plots_tecator[[a + 3]] = fnn_plot
  training_plots_tecator[[a + 4]] = lstm_plot
  training_plots_tecator[[a + 5]] = lstm_bidirectional_plot
  training_plots_tecator[[a + 6]] = gru_plot
  
  
}

# Final Plot
n_plots <- length(training_plots_tecator)
nCol <- 6
do.call("grid.arrange", c(training_plots_tecator, ncol = nCol))

# Running paired t-tests

# Creating data frame
t_test_df = cbind(error_mat_nn[, 1],
                  error_mat_cnn[, 1],
                  error_mat_cnn_2f_32[, 1],
                  error_mat_cnn_3f[, 1],
                  error_mat_cnn_5f[, 1],
                  error_mat_fnn[, 1],
                  error_mat_rnn_lstm[, 1],
                  error_mat_rnn_lstm_bidirectional[, 1],
                  error_mat_rnn_gru[, 1])

# Initializing
p_value_df = matrix(nrow = ncol(t_test_df), ncol = 4)
rownames(p_value_df) = c("NN", "CNN", "CNN; Kernel Size: 2, Filters: 32", "CNN; Kernel Size: 3", "CNN; Kernel Size: 2, Filters: 16", "FNN", "LSTM", "LSTM Bidirectional", "GRU")
colnames(p_value_df) = c("P Value", "T Value", "Lower Bound", "Upper Upper Bound")

# Getting p-values
for(i in 1:ncol(t_test_df)) {
  
  # Selecting data sets
  FNN_ttest = t_test_df[, 6]
  Other_ttest = t_test_df[, i]
  
  # Calculating difference
  d = Other_ttest - FNN_ttest
  
  # Mean difference
  mean_d = mean(d)
  
  # SE
  se_d = sd(d)/sqrt(length(FNN_ttest))
  
  # T value
  T_value = mean_d/se_d
  
  # df
  df_val = length(FNN_ttest) - 1
  
  # p-value
  p_value = pt(abs(T_value), df_val, lower.tail = F)
  
  # Storing
  p_value_df[i, 1] = p_value
  p_value_df[i, 2] = T_value
  p_value_df[i, 3] = mean_d - 1.96*se_d
  p_value_df[i, 4] = mean_d + 1.96*se_d
}

