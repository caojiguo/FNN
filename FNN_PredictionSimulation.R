##############################
#                            #
# FNN Simulations - Final    #
#                            #
##############################

##### Libraries #####
source("FNN.R")

#############################################################
# 1 - Identity 
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1)
use_session_with_seed(
  1,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Number of sims
sim_num = 100

# Initializing matrices for results
error_mat_lm_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_cnn_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_nn_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_lm1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB1_nf = matrix(nrow = sim_num, ncol = 1)
# nn_training_plot <- list()
# cnn_training_plot <- list()
# fnn_training_plot <- list()

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################
  
  # Testing
  # u = 1
  
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
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
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
      alpha[i]
    
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
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))
  
  #########################################
  # Setting up data for functional models #
  #########################################
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running usual functional models #
  ###################################
  
  # Functional Linear Model (Basis)
  l=2^(-4:10)
  func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                               lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  pred_basis = predict(func_basis[[1]], test_x)
  
  # Functional Principal Component Regression (No Penalty)
  func_pc = fregre.pc.cv(train_x, train_y, 8)
  pred_pc = predict(func_pc$fregre.pc, test_x)
  
  # Functional Principal Component Regression (2nd Deriv Penalization)
  func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
  pred_pc2 = predict(func_pc2$fregre.pc, test_x)
  
  # Functional Principal Component Regression (Ridge Regression)
  func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
  pred_pc3 = predict(func_pc3$fregre.pc, test_x)
  
  # Functional Partial Least Squares Regression (No Penalty)
  func_pls = fregre.pls(train_x, train_y, 1:6)
  pred_pls = predict(func_pls, test_x)
  
  # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda = 1:3, P=c(0,0,1))
  pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = predict(func_np, test_x)
  
  
  #############################################
  # Setting up data for non-functional models #
  #############################################
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running regression models       #
  ###################################
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                          newdata = test_f, 
                                          n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = FALSE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
  ########################################
  # Neural Network Tuning Setup          #
  ########################################
  
  # Initializing
  min_error_nn = 99999
  min_error_cnn = 99999
  
  # Setting up MV data
  MV_train = as.data.frame(sim_data[-ind,])
  MV_test = as.data.frame(sim_data[ind,])
  
  # Random Split
  train_split = sample(1:nrow(MV_train), floor(0.75*nrow(MV_train)))
  
  # Learn rates grid
  num_initalizations = 10
  
  ########################################
  # Running Convolutional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_cnn <- keras_model_sequential()
    model_cnn %>% 
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn <- model_cnn %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 250,  
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
      # cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn = mean((c(pred_cnn) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn = error_cnn_train
      
    }
    
  }
  
  ########################################
  # Running Conventional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up NN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_nn <- keras_model_sequential()
    model_nn %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_nn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Training FNN model
    history_nn <- model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                                   train_y[train_split], 
                                   epochs = 250,  
                                   validation_split = 0.2,
                                   callbacks = list(early_stop),
                                   verbose = 0)
    
    # Predictions
    test_predictions <- model_nn %>% predict(as.matrix(MV_train[-train_split,]))
    
    # Plotting
    error_nn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_nn_train < min_error_nn){
      
      # Predictions
      pred_nn <- model_nn %>% predict(as.matrix(MV_test))
      
      # Error
      error_nn = mean((c(pred_nn) - test_y)^2, na.rm = T)
      
      # Saving training plots
      # nn_training_plot[[i]] = as.data.frame(history_nn)
      
      # New Min Error
      min_error_nn = error_nn_train
      
    }
    
  }
  
  
  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Running FNN for simulation
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
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
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm_1[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_1[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_1[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_1[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_1[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_1[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_1[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_cnn_1[u, 1] = mean((c(pred_cnn) - test_y)^2, na.rm = T)
  error_mat_nn_1[u, 1] = mean((c(pred_nn) - test_y)^2, na.rm = T)
  error_mat_fnn_1[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  error_mat_lm1_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin1_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se1_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF1_nf[u, 1] = MSPE_rf
  error_mat_GBM1_nf[u, 1] = MSPE_gbm
  error_mat_PPR1_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS1_nf[u, 1] = MSPE_best_mars
  error_mat_XGB1_nf[u, 1] = MSPE_xgb
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim1 = matrix(nrow = 18, ncol = 2)

# Collecting errors
Final_Table_Sim1[1, 1] = colMeans(error_mat_lm_1, na.rm = T)
Final_Table_Sim1[2, 1] = colMeans(error_mat_np_1, na.rm = T)
Final_Table_Sim1[3, 1] = colMeans(error_mat_pc1_1, na.rm = T)
Final_Table_Sim1[4, 1] = colMeans(error_mat_pc2_1, na.rm = T)
Final_Table_Sim1[5, 1] = colMeans(error_mat_pc3_1, na.rm = T)
Final_Table_Sim1[6, 1] = colMeans(error_mat_pls1_1, na.rm = T)
Final_Table_Sim1[7, 1] = colMeans(error_mat_pls2_1, na.rm = T)
Final_Table_Sim1[8, 1] = colMeans(error_mat_cnn_1, na.rm = T)
Final_Table_Sim1[9, 1] = colMeans(error_mat_nn_1, na.rm = T)
Final_Table_Sim1[10, 1] = colMeans(error_mat_fnn_1, na.rm = T)
Final_Table_Sim1[11, 1] = colMeans(error_mat_lm1_nf, na.rm = T)
Final_Table_Sim1[12, 1] = colMeans(error_mat_lassoMin1_nf, na.rm = T)
Final_Table_Sim1[13, 1] = colMeans(error_mat_lasso1se1_nf, na.rm = T)
Final_Table_Sim1[14, 1] = colMeans(error_mat_RF1_nf, na.rm = T)
Final_Table_Sim1[15, 1] = colMeans(error_mat_GBM1_nf, na.rm = T)
Final_Table_Sim1[16, 1] = colMeans(error_mat_PPR1_nf, na.rm = T)
Final_Table_Sim1[17, 1] = colMeans(error_mat_MARS1_nf, na.rm = T)
Final_Table_Sim1[18, 1] = colMeans(error_mat_XGB1_nf, na.rm = T)

# Collecting SDs
Final_Table_Sim1[1, 2] = colSds(error_mat_lm_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[2, 2] = colSds(error_mat_np_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[3, 2] = colSds(error_mat_pc1_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[4, 2] = colSds(error_mat_pc2_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[5, 2] = colSds(error_mat_pc3_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[6, 2] = colSds(error_mat_pls1_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[7, 2] = colSds(error_mat_pls2_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[8, 2] = colSds(error_mat_cnn_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[9, 2] = colSds(error_mat_nn_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[10, 2] = colSds(error_mat_fnn_1, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[11, 2] = colSds(error_mat_lm1_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[12, 2] = colSds(error_mat_lassoMin1_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[13, 2] = colSds(error_mat_lasso1se1_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[14, 2] = colSds(error_mat_RF1_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[15, 2] = colSds(error_mat_GBM1_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[16, 2] = colSds(error_mat_PPR1_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[17, 2] = colSds(error_mat_MARS1_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim1[18, 2] = colSds(error_mat_XGB1_nf, na.rm = T)/sqrt(sim_num)

# Looking at results
colnames(Final_Table_Sim1) = c("Mean", "SE")
rownames(Final_Table_Sim1) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                               "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
Final_Table_Sim1

# Creating data frame
t_test_df = cbind(error_mat_lm_1[, 1],
                  error_mat_np_1[, 1],
                  error_mat_pc1_1[, 1],
                  error_mat_pc2_1[, 1],
                  error_mat_pc3_1[, 1],
                  error_mat_pls1_1[, 1],
                  error_mat_pls2_1[, 1],
                  error_mat_cnn_1[, 1],
                  error_mat_nn_1[, 1],
                  error_mat_fnn_1[, 1],
                  error_mat_lm1_nf[, 1],
                  error_mat_lassoMin1_nf[, 1],
                  error_mat_lasso1se1_nf[, 1],
                  error_mat_RF1_nf[, 1],
                  error_mat_GBM1_nf[, 1],
                  error_mat_PPR1_nf[, 1],
                  error_mat_MARS1_nf[, 1],
                  error_mat_XGB1_nf[, 1])

# Initializing
p_value_df_sim1 = matrix(nrow = ncol(t_test_df), ncol = 4)
rownames(p_value_df_sim1) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                         "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
colnames(p_value_df_sim1) = c("P Value", "T Value", "Lower Bound", "Upper Bound")

# Getting p-values
for(i in 1:ncol(t_test_df)) {
  
  # Selecting data sets
  FNN_ttest = t_test_df[, 10]
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
  p_value_df_sim1[i, 1] = p_value
  p_value_df_sim1[i, 2] = T_value
  p_value_df_sim1[i, 3] = mean_d - 1.96*se_d
  p_value_df_sim1[i, 4] = mean_d + 1.96*se_d
}


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#############################################################
# 2 - Exponential
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1)
use_session_with_seed(
  1,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Initializing matrices for results
error_mat_lm_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_cnn_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_nn_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_lm2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB2_nf = matrix(nrow = sim_num, ncol = 1)

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################
  
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
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
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
                 alpha[i])
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  ystar = c(scale(ystar))
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))
  
  #########################################
  # Setting up data for functional models #
  #########################################
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running usual functional models #
  ###################################
  
  # Functional Linear Model (Basis)
  l=2^(-4:10)
  func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                               lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  pred_basis = predict(func_basis[[1]], test_x)
  
  # Functional Principal Component Regression (No Penalty)
  func_pc = fregre.pc.cv(train_x, train_y, 8)
  pred_pc = predict(func_pc$fregre.pc, test_x)
  
  # Functional Principal Component Regression (2nd Deriv Penalization)
  func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
  pred_pc2 = predict(func_pc2$fregre.pc, test_x)
  
  # Functional Principal Component Regression (Ridge Regression)
  func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
  pred_pc3 = predict(func_pc3$fregre.pc, test_x)
  
  # Functional Partial Least Squares Regression (No Penalty)
  func_pls = fregre.pls(train_x, train_y, 1:6)
  pred_pls = predict(func_pls, test_x)
  
  # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda = 1:3, P=c(0,0,1))
  pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = predict(func_np, test_x)
  
  
  #############################################
  # Setting up data for non-functional models #
  #############################################
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running regression models       #
  ###################################
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                          newdata = test_f, 
                                          n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = FALSE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
  ########################################
  # Neural Network Tuning Setup          #
  ########################################
  
  # Initializing
  min_error_nn = 99999
  min_error_cnn = 99999
  
  # Setting up MV data
  MV_train = as.data.frame(sim_data[-ind,])
  MV_test = as.data.frame(sim_data[ind,])
  
  # Random Split
  train_split = sample(1:nrow(MV_train), floor(0.75*nrow(MV_train)))
  
  # Learn rates grid
  num_initalizations = 10
  
  ########################################
  # Running Convolutional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_cnn <- keras_model_sequential()
    model_cnn %>% 
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn <- model_cnn %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 250,  
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
      # cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn = mean((c(pred_cnn) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn = error_cnn_train
      
    }
    
  }
  
  ########################################
  # Running Conventional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up NN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_nn <- keras_model_sequential()
    model_nn %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_nn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Training FNN model
    history_nn <- model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                                   train_y[train_split], 
                                   epochs = 250,  
                                   validation_split = 0.2,
                                   callbacks = list(early_stop),
                                   verbose = 0)
    
    # Predictions
    test_predictions <- model_nn %>% predict(as.matrix(MV_train[-train_split,]))
    
    # Plotting
    error_nn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_nn_train < min_error_nn){
      
      # Predictions
      pred_nn <- model_nn %>% predict(as.matrix(MV_test))
      
      # Error
      error_nn = mean((c(pred_nn) - test_y)^2, na.rm = T)
      
      # Saving training plots
      # nn_training_plot[[i]] = as.data.frame(history_nn)
      
      # New Min Error
      min_error_nn = error_nn_train
      
    }
    
  }
  
  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Running FNN for simulation
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
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
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm_2[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_2[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_2[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_2[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_2[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_2[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_2[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_cnn_2[u, 1] = mean((c(pred_cnn) - test_y)^2, na.rm = T)
  error_mat_nn_2[u, 1] = mean((c(pred_nn) - test_y)^2, na.rm = T)
  error_mat_fnn_2[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  error_mat_lm2_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin2_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se2_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF2_nf[u, 1] = MSPE_rf
  error_mat_GBM2_nf[u, 1] = MSPE_gbm
  error_mat_PPR2_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS2_nf[u, 1] = MSPE_best_mars
  error_mat_XGB2_nf[u, 1] = MSPE_xgb
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim2 = matrix(nrow = 18, ncol = 2)

# Collecting errors
Final_Table_Sim2[1, 1] = colMeans(error_mat_lm_2, na.rm = T)
Final_Table_Sim2[2, 1] = colMeans(error_mat_np_2, na.rm = T)
Final_Table_Sim2[3, 1] = colMeans(error_mat_pc1_2, na.rm = T)
Final_Table_Sim2[4, 1] = colMeans(error_mat_pc2_2, na.rm = T)
Final_Table_Sim2[5, 1] = colMeans(error_mat_pc3_2, na.rm = T)
Final_Table_Sim2[6, 1] = colMeans(error_mat_pls1_2, na.rm = T)
Final_Table_Sim2[7, 1] = colMeans(error_mat_pls2_2, na.rm = T)
Final_Table_Sim2[8, 1] = colMeans(error_mat_cnn_2, na.rm = T)
Final_Table_Sim2[9, 1] = colMeans(error_mat_nn_2, na.rm = T)
Final_Table_Sim2[10, 1] = colMeans(error_mat_fnn_2, na.rm = T)
Final_Table_Sim2[11, 1] = colMeans(error_mat_lm2_nf, na.rm = T)
Final_Table_Sim2[12, 1] = colMeans(error_mat_lassoMin2_nf, na.rm = T)
Final_Table_Sim2[13, 1] = colMeans(error_mat_lasso1se2_nf, na.rm = T)
Final_Table_Sim2[14, 1] = colMeans(error_mat_RF2_nf, na.rm = T)
Final_Table_Sim2[15, 1] = colMeans(error_mat_GBM2_nf, na.rm = T)
Final_Table_Sim2[16, 1] = colMeans(error_mat_PPR2_nf, na.rm = T)
Final_Table_Sim2[17, 1] = colMeans(error_mat_MARS2_nf, na.rm = T)
Final_Table_Sim2[18, 1] = colMeans(error_mat_XGB2_nf, na.rm = T)

# Collecting SDs
Final_Table_Sim2[1, 2] = colSds(error_mat_lm_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[2, 2] = colSds(error_mat_np_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[3, 2] = colSds(error_mat_pc1_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[4, 2] = colSds(error_mat_pc2_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[5, 2] = colSds(error_mat_pc3_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[6, 2] = colSds(error_mat_pls1_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[7, 2] = colSds(error_mat_pls2_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[8, 2] = colSds(error_mat_cnn_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[9, 2] = colSds(error_mat_nn_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[10, 2] = colSds(error_mat_fnn_2, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[11, 2] = colSds(error_mat_lm2_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[12, 2] = colSds(error_mat_lassoMin2_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[13, 2] = colSds(error_mat_lasso1se2_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[14, 2] = colSds(error_mat_RF2_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[15, 2] = colSds(error_mat_GBM2_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[16, 2] = colSds(error_mat_PPR2_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[17, 2] = colSds(error_mat_MARS2_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim2[18, 2] = colSds(error_mat_XGB2_nf, na.rm = T)/sqrt(sim_num)

# Looking at results
colnames(Final_Table_Sim2) = c("Mean", "SE")
rownames(Final_Table_Sim2) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                              "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
Final_Table_Sim2

# Creating data frame
t_test_df = cbind(error_mat_lm_2[, 1],
                  error_mat_np_2[, 1],
                  error_mat_pc1_2[, 1],
                  error_mat_pc2_2[, 1],
                  error_mat_pc3_2[, 1],
                  error_mat_pls1_2[, 1],
                  error_mat_pls2_2[, 1],
                  error_mat_cnn_2[, 1],
                  error_mat_nn_2[, 1],
                  error_mat_fnn_2[, 1],
                  error_mat_lm2_nf[, 1],
                  error_mat_lassoMin2_nf[, 1],
                  error_mat_lasso1se2_nf[, 1],
                  error_mat_RF2_nf[, 1],
                  error_mat_GBM2_nf[, 1],
                  error_mat_PPR2_nf[, 1],
                  error_mat_MARS2_nf[, 1],
                  error_mat_XGB2_nf[, 1])

# Initializing
p_value_df_sim2 = matrix(nrow = ncol(t_test_df), ncol = 4)
rownames(p_value_df_sim2) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                              "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
colnames(p_value_df_sim2) = c("P Value", "T Value", "Lower Bound", "Upper Bound")

# Getting p-values
for(i in 1:ncol(t_test_df)) {
  
  # Selecting data sets
  FNN_ttest = t_test_df[, 10]
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
  p_value_df_sim2[i, 1] = p_value
  p_value_df_sim2[i, 2] = T_value
  p_value_df_sim2[i, 3] = mean_d - 1.96*se_d
  p_value_df_sim2[i, 4] = mean_d + 1.96*se_d
}


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#############################################################
# 3 - Sigmoidal
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1)
use_session_with_seed(
  1,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Initializing matrices for results
error_mat_lm_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_cnn_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_nn_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_lm3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB3_nf = matrix(nrow = sim_num, ncol = 1)

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################
  
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
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
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
                        alpha[i]))
    
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
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))
  
  #########################################
  # Setting up data for functional models #
  #########################################
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running usual functional models #
  ###################################
  
  # Functional Linear Model (Basis)
  l=2^(-4:10)
  func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                               lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  pred_basis = predict(func_basis[[1]], test_x)
  
  # Functional Principal Component Regression (No Penalty)
  func_pc = fregre.pc.cv(train_x, train_y, 8)
  pred_pc = predict(func_pc$fregre.pc, test_x)
  
  # Functional Principal Component Regression (2nd Deriv Penalization)
  func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
  pred_pc2 = predict(func_pc2$fregre.pc, test_x)
  
  # Functional Principal Component Regression (Ridge Regression)
  func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
  pred_pc3 = predict(func_pc3$fregre.pc, test_x)
  
  # Functional Partial Least Squares Regression (No Penalty)
  func_pls = fregre.pls(train_x, train_y, 1:6)
  pred_pls = predict(func_pls, test_x)
  
  # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda = 1:3, P=c(0,0,1))
  pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = predict(func_np, test_x)
  
  
  #############################################
  # Setting up data for non-functional models #
  #############################################
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running regression models       #
  ###################################
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                          newdata = test_f, 
                                          n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = FALSE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
  ########################################
  # Neural Network Tuning Setup          #
  ########################################
  
  # Initializing
  min_error_nn = 99999
  min_error_cnn = 99999
  
  # Setting up MV data
  MV_train = as.data.frame(sim_data[-ind,])
  MV_test = as.data.frame(sim_data[ind,])
  
  # Random Split
  train_split = sample(1:nrow(MV_train), floor(0.75*nrow(MV_train)))
  
  # Learn rates grid
  num_initalizations = 10
  
  ########################################
  # Running Convolutional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_cnn <- keras_model_sequential()
    model_cnn %>% 
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn <- model_cnn %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 250,  
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
      # cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn = mean((c(pred_cnn) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn = error_cnn_train
      
    }
    
  }
  
  ########################################
  # Running Conventional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up NN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_nn <- keras_model_sequential()
    model_nn %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_nn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Training FNN model
    history_nn <- model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                                   train_y[train_split], 
                                   epochs = 250,  
                                   validation_split = 0.2,
                                   callbacks = list(early_stop),
                                   verbose = 0)
    
    # Predictions
    test_predictions <- model_nn %>% predict(as.matrix(MV_train[-train_split,]))
    
    # Plotting
    error_nn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_nn_train < min_error_nn){
      
      # Predictions
      pred_nn <- model_nn %>% predict(as.matrix(MV_test))
      
      # Error
      error_nn = mean((c(pred_nn) - test_y)^2, na.rm = T)
      
      # Saving training plots
      # nn_training_plot[[i]] = as.data.frame(history_nn)
      
      # New Min Error
      min_error_nn = error_nn_train
      
    }
    
  }
  
  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Running FNN for simulation
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
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
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm_3[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_3[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_3[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_3[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_3[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_3[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_3[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_cnn_3[u, 1] = mean((c(pred_cnn) - test_y)^2, na.rm = T)
  error_mat_nn_3[u, 1] = mean((c(pred_nn) - test_y)^2, na.rm = T)
  error_mat_fnn_3[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  error_mat_lm3_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin3_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se3_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF3_nf[u, 1] = MSPE_rf
  error_mat_GBM3_nf[u, 1] = MSPE_gbm
  error_mat_PPR3_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS3_nf[u, 1] = MSPE_best_mars
  error_mat_XGB3_nf[u, 1] = MSPE_xgb
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim3 = matrix(nrow = 18, ncol = 2)

# Collecting errors
Final_Table_Sim3[1, 1] = colMeans(error_mat_lm_3, na.rm = T)
Final_Table_Sim3[2, 1] = colMeans(error_mat_np_3, na.rm = T)
Final_Table_Sim3[3, 1] = colMeans(error_mat_pc1_3, na.rm = T)
Final_Table_Sim3[4, 1] = colMeans(error_mat_pc2_3, na.rm = T)
Final_Table_Sim3[5, 1] = colMeans(error_mat_pc3_3, na.rm = T)
Final_Table_Sim3[6, 1] = colMeans(error_mat_pls1_3, na.rm = T)
Final_Table_Sim3[7, 1] = colMeans(error_mat_pls2_3, na.rm = T)
Final_Table_Sim3[8, 1] = colMeans(error_mat_cnn_3, na.rm = T)
Final_Table_Sim3[9, 1] = colMeans(error_mat_nn_3, na.rm = T)
Final_Table_Sim3[10, 1] = colMeans(error_mat_fnn_3, na.rm = T)
Final_Table_Sim3[11, 1] = colMeans(error_mat_lm3_nf, na.rm = T)
Final_Table_Sim3[12, 1] = colMeans(error_mat_lassoMin3_nf, na.rm = T)
Final_Table_Sim3[13, 1] = colMeans(error_mat_lasso1se3_nf, na.rm = T)
Final_Table_Sim3[14, 1] = colMeans(error_mat_RF3_nf, na.rm = T)
Final_Table_Sim3[15, 1] = colMeans(error_mat_GBM3_nf, na.rm = T)
Final_Table_Sim3[16, 1] = colMeans(error_mat_PPR3_nf, na.rm = T)
Final_Table_Sim3[17, 1] = colMeans(error_mat_MARS3_nf, na.rm = T)
Final_Table_Sim3[18, 1] = colMeans(error_mat_XGB3_nf, na.rm = T)

# Collecting SDs
Final_Table_Sim3[1, 2] = colSds(error_mat_lm_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[2, 2] = colSds(error_mat_np_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[3, 2] = colSds(error_mat_pc1_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[4, 2] = colSds(error_mat_pc2_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[5, 2] = colSds(error_mat_pc3_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[6, 2] = colSds(error_mat_pls1_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[7, 2] = colSds(error_mat_pls2_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[8, 2] = colSds(error_mat_cnn_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[9, 2] = colSds(error_mat_nn_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[10, 2] = colSds(error_mat_fnn_3, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[11, 2] = colSds(error_mat_lm3_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[12, 2] = colSds(error_mat_lassoMin3_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[13, 2] = colSds(error_mat_lasso1se3_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[14, 2] = colSds(error_mat_RF3_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[15, 2] = colSds(error_mat_GBM3_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[16, 2] = colSds(error_mat_PPR3_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[17, 2] = colSds(error_mat_MARS3_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim3[18, 2] = colSds(error_mat_XGB3_nf, na.rm = T)/sqrt(sim_num)

# Looking at results
colnames(Final_Table_Sim3) = c("Mean", "SE")
rownames(Final_Table_Sim3) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                               "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
Final_Table_Sim3


# Creating data frame
t_test_df = cbind(error_mat_lm_3[, 1],
                  error_mat_np_3[, 1],
                  error_mat_pc1_3[, 1],
                  error_mat_pc2_3[, 1],
                  error_mat_pc3_3[, 1],
                  error_mat_pls1_3[, 1],
                  error_mat_pls2_3[, 1],
                  error_mat_cnn_3[, 1],
                  error_mat_nn_3[, 1],
                  error_mat_fnn_3[, 1],
                  error_mat_lm3_nf[, 1],
                  error_mat_lassoMin3_nf[, 1],
                  error_mat_lasso1se3_nf[, 1],
                  error_mat_RF3_nf[, 1],
                  error_mat_GBM3_nf[, 1],
                  error_mat_PPR3_nf[, 1],
                  error_mat_MARS3_nf[, 1],
                  error_mat_XGB3_nf[, 1])

# Initializing
p_value_df_sim3 = matrix(nrow = ncol(t_test_df), ncol = 4)
rownames(p_value_df_sim3) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                              "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
colnames(p_value_df_sim3) = c("P Value", "T Value", "Lower Bound", "Upper Bound")

# Getting p-values
for(i in 1:ncol(t_test_df)) {
  
  # Selecting data sets
  FNN_ttest = t_test_df[, 10]
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
  p_value_df_sim3[i, 1] = p_value
  p_value_df_sim3[i, 2] = T_value
  p_value_df_sim3[i, 3] = mean_d - 1.96*se_d
  p_value_df_sim3[i, 4] = mean_d + 1.96*se_d
}


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#############################################################
# 4 - Log
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1)
use_session_with_seed(
  1,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Initializing matrices for results
error_mat_lm_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_cnn_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_nn_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_lm4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB4_nf = matrix(nrow = sim_num, ncol = 1)

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################
  
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
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
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
                 alpha[i])
    
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
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))
  
  #########################################
  # Setting up data for functional models #
  #########################################
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running usual functional models #
  ###################################
  
  # Functional Linear Model (Basis)
  l=2^(-4:10)
  func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                               lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  pred_basis = predict(func_basis[[1]], test_x)
  
  # Functional Principal Component Regression (No Penalty)
  func_pc = fregre.pc.cv(train_x, train_y, 8)
  pred_pc = predict(func_pc$fregre.pc, test_x)
  
  # Functional Principal Component Regression (2nd Deriv Penalization)
  func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
  pred_pc2 = predict(func_pc2$fregre.pc, test_x)
  
  # Functional Principal Component Regression (Ridge Regression)
  func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
  pred_pc3 = predict(func_pc3$fregre.pc, test_x)
  
  # Functional Partial Least Squares Regression (No Penalty)
  func_pls = fregre.pls(train_x, train_y, 1:6)
  pred_pls = predict(func_pls, test_x)
  
  # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda = 1:3, P=c(0,0,1))
  pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = predict(func_np, test_x)
  
  
  #############################################
  # Setting up data for non-functional models #
  #############################################
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
  ###################################
  # Running regression models       #
  ###################################
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                          newdata = test_f, 
                                          n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = FALSE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
  ########################################
  # Neural Network Tuning Setup          #
  ########################################
  
  # Initializing
  min_error_nn = 99999
  min_error_cnn = 99999
  
  # Setting up MV data
  MV_train = as.data.frame(sim_data[-ind,])
  MV_test = as.data.frame(sim_data[ind,])
  
  # Random Split
  train_split = sample(1:nrow(MV_train), floor(0.75*nrow(MV_train)))
  
  # Learn rates grid
  num_initalizations = 10
  
  ########################################
  # Running Convolutional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up CNN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_cnn <- keras_model_sequential()
    model_cnn %>% 
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_cnn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn <- model_cnn %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 250,  
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
      # cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn = mean((c(pred_cnn) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn = error_cnn_train
      
    }
    
  }
  
  ########################################
  # Running Conventional Neural Network #
  ########################################
  
  # Setting seeds
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up NN model
  for(i in 1:num_initalizations){
    
    # setting up model
    model_nn <- keras_model_sequential()
    model_nn %>% 
      layer_dense(units = 16, activation = 'relu') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 16, activation = 'linear') %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Setting parameters for NN model
    model_nn %>% compile(
      optimizer = optimizer_adam(lr = 0.001), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)
    
    # Training FNN model
    history_nn <- model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                                   train_y[train_split], 
                                   epochs = 250,  
                                   validation_split = 0.2,
                                   callbacks = list(early_stop),
                                   verbose = 0)
    
    # Predictions
    test_predictions <- model_nn %>% predict(as.matrix(MV_train[-train_split,]))
    
    # Plotting
    error_nn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_nn_train < min_error_nn){
      
      # Predictions
      pred_nn <- model_nn %>% predict(as.matrix(MV_test))
      
      # Error
      error_nn = mean((c(pred_nn) - test_y)^2, na.rm = T)
      
      # Saving training plots
      # nn_training_plot[[i]] = as.data.frame(history_nn)
      
      # New Min Error
      min_error_nn = error_nn_train
      
    }
    
  }
  
  # Running FNN for simulation
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
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
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm_4[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_4[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_4[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_4[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_4[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_4[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_4[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_cnn_4[u, 1] = mean((c(pred_cnn) - test_y)^2, na.rm = T)
  error_mat_nn_4[u, 1] = mean((c(pred_nn) - test_y)^2, na.rm = T)
  error_mat_fnn_4[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  error_mat_lm4_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin4_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se4_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF4_nf[u, 1] = MSPE_rf
  error_mat_GBM4_nf[u, 1] = MSPE_gbm
  error_mat_PPR4_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS4_nf[u, 1] = MSPE_best_mars
  error_mat_XGB4_nf[u, 1] = MSPE_xgb
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim4 = matrix(nrow = 18, ncol = 2)

# Collecting errors
Final_Table_Sim4[1, 1] = colMeans(error_mat_lm_4, na.rm = T)
Final_Table_Sim4[2, 1] = colMeans(error_mat_np_4, na.rm = T)
Final_Table_Sim4[3, 1] = colMeans(error_mat_pc1_4, na.rm = T)
Final_Table_Sim4[4, 1] = colMeans(error_mat_pc2_4, na.rm = T)
Final_Table_Sim4[5, 1] = colMeans(error_mat_pc3_4, na.rm = T)
Final_Table_Sim4[6, 1] = colMeans(error_mat_pls1_4, na.rm = T)
Final_Table_Sim4[7, 1] = colMeans(error_mat_pls2_4, na.rm = T)
Final_Table_Sim4[8, 1] = colMeans(error_mat_cnn_4, na.rm = T)
Final_Table_Sim4[9, 1] = colMeans(error_mat_nn_4, na.rm = T)
Final_Table_Sim4[10, 1] = colMeans(error_mat_fnn_4, na.rm = T)
Final_Table_Sim4[11, 1] = colMeans(error_mat_lm4_nf, na.rm = T)
Final_Table_Sim4[12, 1] = colMeans(error_mat_lassoMin4_nf, na.rm = T)
Final_Table_Sim4[13, 1] = colMeans(error_mat_lasso1se4_nf, na.rm = T)
Final_Table_Sim4[14, 1] = colMeans(error_mat_RF4_nf, na.rm = T)
Final_Table_Sim4[15, 1] = colMeans(error_mat_GBM4_nf, na.rm = T)
Final_Table_Sim4[16, 1] = colMeans(error_mat_PPR4_nf, na.rm = T)
Final_Table_Sim4[17, 1] = colMeans(error_mat_MARS4_nf, na.rm = T)
Final_Table_Sim4[18, 1] = colMeans(error_mat_XGB4_nf, na.rm = T)

# Collecting SDs
Final_Table_Sim4[1, 2] = colSds(error_mat_lm_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[2, 2] = colSds(error_mat_np_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[3, 2] = colSds(error_mat_pc1_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[4, 2] = colSds(error_mat_pc2_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[5, 2] = colSds(error_mat_pc3_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[6, 2] = colSds(error_mat_pls1_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[7, 2] = colSds(error_mat_pls2_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[8, 2] = colSds(error_mat_cnn_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[9, 2] = colSds(error_mat_nn_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[10, 2] = colSds(error_mat_fnn_4, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[11, 2] = colSds(error_mat_lm4_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[12, 2] = colSds(error_mat_lassoMin4_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[13, 2] = colSds(error_mat_lasso1se4_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[14, 2] = colSds(error_mat_RF4_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[15, 2] = colSds(error_mat_GBM4_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[16, 2] = colSds(error_mat_PPR4_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[17, 2] = colSds(error_mat_MARS4_nf, na.rm = T)/sqrt(sim_num)
Final_Table_Sim4[18, 2] = colSds(error_mat_XGB4_nf, na.rm = T)/sqrt(sim_num)

# Looking at results
colnames(Final_Table_Sim4) = c("Mean", "SE")
rownames(Final_Table_Sim4) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                              "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
Final_Table_Sim4

# Creating data frame
t_test_df = cbind(error_mat_lm_4[, 1],
                  error_mat_np_4[, 1],
                  error_mat_pc1_4[, 1],
                  error_mat_pc2_4[, 1],
                  error_mat_pc3_4[, 1],
                  error_mat_pls1_4[, 1],
                  error_mat_pls2_4[, 1],
                  error_mat_cnn_4[, 1],
                  error_mat_nn_4[, 1],
                  error_mat_fnn_4[, 1],
                  error_mat_lm4_nf[, 1],
                  error_mat_lassoMin4_nf[, 1],
                  error_mat_lasso1se4_nf[, 1],
                  error_mat_RF4_nf[, 1],
                  error_mat_GBM4_nf[, 1],
                  error_mat_PPR4_nf[, 1],
                  error_mat_MARS4_nf[, 1],
                  error_mat_XGB4_nf[, 1])

# Initializing
p_value_df_sim4 = matrix(nrow = ncol(t_test_df), ncol = 4)
rownames(p_value_df_sim4) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN",
                              "LM", "LASSO Min", "LASSO 1se", "RF", "GBM", "PPR", "MARS", "XGB")
colnames(p_value_df_sim4) = c("P Value", "T Value", "Lower Bound", "Upper Bound")

# Getting p-values
for(i in 1:ncol(t_test_df)) {
  
  # Selecting data sets
  FNN_ttest = t_test_df[, 10]
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
  p_value_df_sim4[i, 1] = p_value
  p_value_df_sim4[i, 2] = T_value
  p_value_df_sim4[i, 3] = mean_d - 1.96*se_d
  p_value_df_sim4[i, 4] = mean_d + 1.96*se_d
}

################################################################################
# Creating MSPE Plots # #
################################################################################

# Making matrices for each simulation
sim1_mat = cbind(error_mat_lm1_nf, 
                 error_mat_lassoMin1_nf,
                 error_mat_lasso1se1_nf,
                 error_mat_RF1_nf,
                 error_mat_GBM1_nf,
                 error_mat_PPR1_nf,
                 error_mat_XGB1_nf,
                 error_mat_lm_1,
                 error_mat_pc1_1,
                 error_mat_pc2_1,
                 error_mat_pc3_1,
                 error_mat_pls1_1,
                 error_mat_pls2_1,
                 error_mat_cnn_1,
                 error_mat_nn_1,
                 error_mat_fnn_1)

sim2_mat = cbind(error_mat_lm2_nf, 
                 error_mat_lassoMin2_nf,
                 error_mat_lasso1se2_nf,
                 error_mat_RF2_nf,
                 error_mat_GBM2_nf,
                 error_mat_PPR2_nf,
                 error_mat_XGB2_nf,
                 error_mat_lm_2,
                 error_mat_pc1_2,
                 error_mat_pc2_2,
                 error_mat_pc3_2,
                 error_mat_pls1_2,
                 error_mat_pls2_2,
                 error_mat_cnn_2,
                 error_mat_nn_2,
                 error_mat_fnn_2)

sim3_mat = cbind(error_mat_lm3_nf, 
                 error_mat_lassoMin3_nf,
                 error_mat_lasso1se3_nf,
                 error_mat_RF3_nf,
                 error_mat_GBM3_nf,
                 error_mat_PPR3_nf,
                 error_mat_XGB3_nf,
                 error_mat_lm_3,
                 error_mat_pc1_3,
                 error_mat_pc2_3,
                 error_mat_pc3_3,
                 error_mat_pls1_3,
                 error_mat_pls2_3,
                 error_mat_cnn_3,
                 error_mat_nn_3,
                 error_mat_fnn_3)

sim4_mat = cbind(error_mat_lm4_nf, 
                 error_mat_lassoMin4_nf,
                 error_mat_lasso1se4_nf,
                 error_mat_RF4_nf,
                 error_mat_GBM4_nf,
                 error_mat_PPR4_nf,
                 error_mat_XGB4_nf,
                 error_mat_lm_4,
                 error_mat_pc1_4,
                 error_mat_pc2_4,
                 error_mat_pc3_4,
                 error_mat_pls1_4,
                 error_mat_pls2_4,
                 error_mat_cnn_4,
                 error_mat_nn_4,
                 error_mat_fnn_4)

# Names
names_list = c("MLR", "LASSO_Min", "LASOO_1se", "RF", "GBM", "PPR", "XGB", 
               "FLM", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN")

# Changing
colnames(sim1_mat) = names_list
colnames(sim2_mat) = names_list
colnames(sim3_mat) = names_list
colnames(sim4_mat) = names_list

# Saving matrices
write.table(sim1_mat, file="sim1Pred.csv", row.names = F)
write.table(sim2_mat, file="sim2Pred.csv", row.names = F)
write.table(sim3_mat, file="sim3Pred.csv", row.names = F)
write.table(sim4_mat, file="sim4Pred.csv", row.names = F)


# Creating boxplots

#### Sim 1

# Saving sqrt
sqrt_MSPE <- data.frame(sqrt(sim1_mat))

# Creating boxplots
plot1 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkred") + 
  theme_bw() + 
  xlab("Model\nSimulation: 1") +
  ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 1)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot1

#### Sim 2

# Saving sqrt
sqrt_MSPE <- data.frame(sim2_mat)

# Creating boxplots
plot2 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkred") + 
  theme_bw() + 
  xlab("Model\nSimulation: 2") +
  ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot2

#### Sim 3

# Saving sqrt
sqrt_MSPE <- data.frame(sqrt(sim3_mat))

# Creating boxplots
plot3 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkred") + 
  theme_bw() + 
  xlab("Model\nSimulation: 3") +
  ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 0.2)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot3

#### Sim 4

# Saving sqrt
sqrt_MSPE <- data.frame(sqrt(sim4_mat))

# Creating boxplots
plot4 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkred") + 
  theme_bw() + 
  xlab("Model\nSimulation: 4") +
  ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 3)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot4


###################### RELATIVE PLOTS ##########################

# Getting minimums
mspe_div1_mins = apply(sim1_mat, 1, function(x){
  return(min(x))
})

mspe_div2_mins = apply(sim2_mat, 1, function(x){
  return(min(x))
})

mspe_div3_mins = apply(sim3_mat, 1, function(x){
  return(min(x))
})

mspe_div4_mins = apply(sim4_mat, 1, function(x){
  return(min(x))
})

# Initializing
mspe_div1 = matrix(nrow = nrow(sim1_mat), ncol = ncol(sim1_mat))
mspe_div2 = matrix(nrow = nrow(sim2_mat), ncol = ncol(sim2_mat))
mspe_div3 = matrix(nrow = nrow(sim3_mat), ncol = ncol(sim3_mat))
mspe_div4 = matrix(nrow = nrow(sim4_mat), ncol = ncol(sim4_mat))

for (i in 1:sim_num) {
  mspe_div1[i, ] = sim1_mat[i,]/mspe_div1_mins[i]
  mspe_div2[i, ] = sim2_mat[i,]/mspe_div2_mins[i]
  mspe_div3[i, ] = sim3_mat[i,]/mspe_div3_mins[i]
  mspe_div4[i, ] = sim4_mat[i,]/mspe_div4_mins[i]
  
}

# names
colnames(mspe_div1) = names_list
colnames(mspe_div2) = names_list
colnames(mspe_div3) = names_list
colnames(mspe_div4) = names_list

# Creating relative boxplots

# turning into df
df_MSPE <- data.frame(mspe_div1)

# Creating boxplots
plot1_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 1\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme(axis.text.x = element_blank())

plot1_rel

# turning into df
df_MSPE <- data.frame(mspe_div2)

# Creating boxplots
plot2_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 2\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme(axis.text.x = element_blank())

plot2_rel

# turning into df
df_MSPE <- data.frame(mspe_div3)

# Creating boxplots
plot3_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 3\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme(axis.text.x = element_blank())

plot3_rel

# turning into df
df_MSPE <- data.frame(mspe_div4)

# Creating boxplots
plot4_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 4\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed")

plot4_rel

# Saving plots (10 x 13 pdf)
grid.draw(rbind(ggplotGrob(plot1_rel), ggplotGrob(plot2_rel), ggplotGrob(plot3_rel), ggplotGrob(plot4_rel), size = "last")) # pdf 11 x 15


# Check 1