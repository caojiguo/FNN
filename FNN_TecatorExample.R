##############################
#                            #
# Tecator Data Set - Final   #
#                            #
##############################

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
# Parameter Count in FNN: 4029
# Parameter Count in CNN: 87045
# Parameter Count in NN: 5757
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

# Getting data into right format
tecator_data = array(dim = c(nbasis, 215, 1))
tecator_data[,,1] = tecator_deriv2$coefs

# Splitting into test and train
ind = 1:165
tec_data_train <- array(dim = c(nbasis, length(ind), 1))
tec_data_test <- array(dim = c(nbasis, nrow(tecator$absorp.fdata$data) - length(ind), 1))
tec_data_train[,,1] = tecator_data[, ind, ]
tec_data_test[,,1] = tecator_data[, -ind, ]
tecResp_train = tecator_resp[ind]
tecResp_test = tecator_resp[-ind]
scalar_train = data.frame(tecator_scalar[ind,1])
scalar_test = data.frame(tecator_scalar[-ind,1])

# Setting up network
tecator_comp = FNN(resp = tecResp_train, 
                      func_cov = tec_data_train, 
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
                      val_split = 0.15,
                      patience_param = 35,
                      learn_rate = 0.005,
                      decay_rate = 0,
                      batch_size = 32,
                      early_stop = F,
                      print_info = T)

# Predicting
pred_tec = FNN_Predict(tecator_comp,
                       tec_data_test, 
                       scalar_cov = scalar_test,
                       basis_choice = c("fourier"), 
                       num_basis = 3,
                       domain_range = list(c(850, 1050)))

# Getting back results
MEP_FNN = mean(((pred_tec - tecResp_test)^2))/var(tecResp_test)
Rsquared_FNN = 1 - sum((pred_tec - tecResp_test)^2)/sum((tecResp_test - mean(tecResp_test))^2)

### NN Set Up ###

# Initializing
min_error_nn = 99999
min_error_cnn = 99999
nn_training_plot <- list()
cnn_training_plot <- list()

# Setting up MV data
MV_train = as.data.frame(cbind(as.data.frame((tecator$absorp.fdata$data)[ind,]), water = scalar_train))
MV_test = as.data.frame(cbind(as.data.frame((tecator$absorp.fdata$data)[-ind,]), water = scalar_test))

# Random Split
train_split = sample(1:nrow(MV_train), floor(0.8*nrow(MV_train)))

# Initialization Count
num_initalizations = 10
i = 1

### NN

# Setting seeds
set.seed(17)
use_session_with_seed(
  17,
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
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 35)
  
  # Training FNN model
  history_nn <- model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                                 tecResp_train[train_split], 
                                 epochs = 300,  
                                 validation_split = 0.15,
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
    MEP_NN = mean((c(pred_nn) - tecResp_test)^2, na.rm = T)/var(tecResp_test)
    Rsquared_NN = 1 - sum((pred_nn - tecResp_test)^2)/sum((tecResp_test - mean(tecResp_test))^2)
    
    # Saving training plots
    nn_training_plot[[i]] = as.data.frame(history_nn)
    
    # New Min Error
    min_error_nn = error_nn_train
    
  }
  
}


### CNN

# Setting seeds
set.seed(17)
use_session_with_seed(
  17,
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
  
  # Setting up data
  reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
  reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
  reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
  reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
  
  # Early stopping
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 35)
  
  # Training CNN model
  history_cnn <- model_cnn %>% fit(reshaped_data_tensor_train, 
                                   tecResp_train[train_split], 
                                   epochs = 300,  
                                   validation_split = 0.15,
                                   callbacks = list(early_stop),
                                   verbose = 0)
  
  # Predictions
  test_predictions <- model_cnn %>% predict(reshaped_data_tensor_test)
  
  # Plotting
  error_cnn_train = mean((c(test_predictions) - tecResp_train[-train_split])^2)
  
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
    MEP_CNN = mean((c(pred_cnn) - tecResp_test)^2, na.rm = T)/var(tecResp_test)
    Rsquared_CNN = 1 - sum((pred_cnn - tecResp_test)^2)/sum((tecResp_test - mean(tecResp_test))^2)
    
    # New Min Error
    min_error_cnn = error_cnn_train
    
  }
  
}

### Creating Training Plots ###

# Saving relevant
current_cnn = cnn_training_plot[[i]]
current_nn = nn_training_plot[[i]]
current_fnn = data.frame(epoch = 1:length(tecator_comp$per_iter_info$loss), value = tecator_comp$per_iter_info$loss)

# Filtering
current_cnn = current_cnn %>% dplyr::filter(metric == "loss" & data == "validation")
current_nn = current_nn %>% dplyr::filter(metric == "loss" & data == "validation")

cnn_plot = current_cnn %>% 
  ggplot(aes(x = epoch, y = value)) +
  geom_line(size = 1.5,  color='red') + 
  theme_bw() +
  xlab("Epoch") +
  ylab("Validation Loss") +
  xlim(c(0, 1000)) +
  ggtitle(paste("Convolutional Neural Network; Tecator Example")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=12, face = "bold"),
        axis.title=element_text(size=12,face="bold"))

nn_plot = current_nn %>% 
  ggplot(aes(x = epoch, y = value)) +
  geom_line(size = 1.5, color='green') + 
  theme_bw() +
  xlab("Epoch") +
  ylab("Validation Loss") +
  xlim(c(0, 1000)) +
  ggtitle(paste("Neural Network; Tecator Example")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=12, face = "bold"),
        axis.title=element_text(size=12,face="bold"))

fnn_plot = current_fnn %>% 
  ggplot(aes(x = epoch, y = value)) +
  geom_line(size = 1.5, color='blue') + 
  theme_bw() +
  xlab("Epoch") +
  ylab("Validation Loss") +
  xlim(c(0, 1000)) +
  ggtitle(paste("Functional Neural Network;  Tecator Example")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=12, face = "bold"),
        axis.title=element_text(size=12,face="bold"))

# Final Plot
list_plots = list(cnn_plot, nn_plot, fnn_plot)
n_plots <- length(list_plots)
nCol <- 1
do.call("grid.arrange", c(list_plots, ncol = nCol)) # Saved as 10 x 13 PDF file

# Check 1