##################################################
############# Keras Implementation ###############
#### Functional Neural Networks ##################
##################################################

##### Main Functions #####

# (0) Packages

library(keras)
library(tensorflow)
library(fda)
library(ggplot2)
library(ggpubr)
library(caret)
library(pbapply)
library(fda.usc)
library(gridExtra)
library(tidyverse)
library(reshape)
library(randomForest)
library(future.apply)
library(earth)
library(gam)
library(gbm)
library(matrixStats)
library(glmnet)
library(FinCal)
library(reshape2)
library(grid)
library(flux)

# UNCOMMONENT THESE NEXT 3 LINES TO INSTALL 
# require(devtools)
# install_version("tensorflow", version = "2.2.0", repos = "http://cran.us.r-project.org")
# install_version("keras", version = "2.2.5.0", repos = "http://cran.us.r-project.org")

# Setting up python environment (UNCOMMENT THESE AND SET PATH)
# Remember to delete .RData in your working directory before running these as R stores a cache of what Python version to use
# Remember to adjust the code so that the path is pointing towards your python 3.7 environment
library(reticulate)
# use_condaenv(condaenv = 'PFDA', conda = "C:/~path/anaconda3/envs/Python37/python.exe")
# use_python("C:/~path/anaconda3/envs/Python37/python.exe")



# (0) First Layer Function -- for direct connection to Keras

# First Layer Function (Was not used but here for testing purposes)
FNN_First_Layer <- function(func_cov, 
                            scalar_cov = NULL,
                            basis_choice, 
                            num_basis,
                            domain_range,
                            covariate_scaling = T,
                            raw_data = F,
                            print_info = F){

  
  ##### Helper Functions #####
  
  # Integration Approximation for fourier and b-spline
  integral_eval <- function(functional_data,
                            beta_basis,
                            num_fd_basis = dim(func_cov)[1],
                            num_beta_basis,
                            range){
    
    if(beta_basis == "fourier"){
      
      # So, first the basis is created for functional, x(s)
      fourier_basis_feature = create.fourier.basis(rangeval = c(range[1], range[2]),
                                                   nbasis = num_fd_basis)
      
      # evaluating fourier basis for x(s)
      integ_values_fourier = eval.basis(evalarg = seq(range[1], range[2], length.out = 100),
                                        basisobj = fourier_basis_feature)
      
      # functional observation
      x_temp_fourier = functional_data%*%t(integ_values_fourier)
      
      # Now making b spline for beta(s)
      fourier_basis_beta = create.fourier.basis(rangeval = c(range[1], range[2]),
                                                nbasis = num_beta_basis)
      
      # evaluating functions for beta(s)
      fourier_evals = eval.basis(evalarg = seq(range[1], range[2], length.out = 100),
                                 basisobj = fourier_basis_beta)
      
      # Getting x(s)*beta basis function (integrand)
      eval_func_fourier = apply(fourier_evals, 2, function(x){return((x*x_temp_fourier))})
      
      # Getting integral
      integral_fourier = apply(eval_func_fourier, 2, function(x){
        return(auc(x = seq(range[1], range[2], length.out = 100),
                   y = x))})
      
      # returning
      return(integral_fourier)
      
    }
    
    if(beta_basis == "bspline"){
      
      if(num_fd_basis > 3){
        order_chosen_fd = 4
      } else {
        order_chosen_fd = num_fd_basis
      }
      
      if(num_beta_basis > 3){
        order_chosen_beta = 4
      } else {
        order_chosen_beta = num_beta_basis
      }
      
      # So, first the basis is created for functional, x(s)
      bspline_basis_feature = create.bspline.basis(rangeval = c(range[1], range[2]),
                                                   nbasis = num_fd_basis,
                                                   norder = order_chosen_fd)
      
      # evaluating b-spline basis for x(s)
      integ_values_bspline = eval.basis(evalarg = seq(range[1], range[2], length.out = 100),
                                        basisobj = bspline_basis_feature)
      
      # functional observation
      x_temp_bspline = functional_data%*%t(integ_values_bspline)
      
      # Now making b spline for beta(s)
      bspline_basis_beta = create.bspline.basis(rangeval = c(range[1], range[2]),
                                                nbasis = num_beta_basis,
                                                norder = order_chosen_beta)
      
      # evaluating functions for beta(s)
      bspline_evals = eval.basis(evalarg = seq(range[1], range[2], length.out = 100),
                                 basisobj = bspline_basis_beta)
      
      # Getting x(s)*beta basis function (integrand)
      eval_func_bspline = apply(bspline_evals, 2, function(x){return((x*x_temp_bspline))})
      
      # Getting integral
      integral_bspline = apply(eval_func_bspline, 2, function(x){
        return(auc(x = seq(range[1], range[2], length.out = 100),
                   y = x))})
      
      return(integral_bspline)
    }
    
    
  }
  
  if(is.null(scalar_cov)){
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis)))
  } else {
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis) + ncol(scalar_cov)))
  }
  
  # Looping to get approximations
  if(print_info == TRUE){
    message(paste0("Evaluating Integrals:"))
  }
  
  for (i in 1:dim(func_cov)[3]) {
    
    # Current data set
    df <- func_cov[,,i]
    
    # Turning into matrix
    if(is.vector(df) == TRUE){
      # print('yes')
      test_mat = matrix(nrow = length(df), ncol = 1)
      test_mat[,1] = df
      df = test_mat
    }
    
    # Current number of basis and choice of basis information
    cur_basis_num <- num_basis[i]
    cur_basis <- basis_choice[i]
    
    # Getting current range
    cur_range <- domain_range[[i]]
    
    # Storing previous numbers
    if(i == 1){
      left_end = 1
      right_end = cur_basis_num
    } else {
      left_end = sum(num_basis[1:(i - 1)]) + 1
      right_end = (left_end - 1) + cur_basis_num
    }
    
    # Getting evaluations
    if(print_info == TRUE){
      converted_df[, left_end:right_end] = t(pbapply(df, 2, integral_eval, beta_basis = cur_basis,
                                                     num_beta_basis = cur_basis_num,
                                                     range = cur_range))
    } else {
      converted_df[, left_end:right_end] = t(apply(df, 2, integral_eval, beta_basis = cur_basis,
                                                   num_beta_basis = cur_basis_num,
                                                   range = cur_range))
    }
  }
  
  # Now attaching scalar covariates
  if(is.null(scalar_cov)){
    converted_df <- converted_df
  } else{
    for (k in 1:nrow(converted_df)) {
      converted_df[k, (sum(num_basis) + 1):(sum(num_basis) + ncol(scalar_cov))] <- scalar_cov[k,]
    }
  }
  
  # Now we have the data set to pass onto the network, we can set up the data so that it is well suited to be
  # passed onto the network. This means normalizing things and rewriting some other things
  
  # Normalize training data
  if(covariate_scaling == TRUE){
    train_x <- scale(converted_df)
  } else {
    train_x <- as.matrix(cbind(converted_df[,c(1:sum(num_basis))], scale(converted_df[,-c(1:sum(num_basis))])))
  }
  
  # #### Error Checks
  # 
  # if(length(domain_range) != length(num_basis)){
  #   stop("The number of domain ranges doesn't match length of num_basis")
  # }
  # 
  # if(length(domain_range) != length(basis_choice)){
  #   stop("The number of domain ranges doesn't match number of basis choices")
  # }
  # 
  # if(length(num_basis) != length(basis_choice)){
  #   stop("Too many/few num_basis - doesn't match number of basis choices")
  # }
  # 
  # # Getting check for raw vs. non raw
  # if(raw_data == T){
  #   dim_check = length(func_cov)
  # } else {
  #   dim_check = dim(func_cov)[3]
  # }
  # 
  # if(dim_check > length(num_basis)){
  #   
  #   # Fixing domain range
  #   domain_range_list = list()
  #   
  #   for (t in 1:dim_check) {
  #     
  #     domain_range_list[[t]] = domain_range[[1]]
  #     
  #   }
  #   
  #   # Fixing num basis
  #   num_basis = rep(num_basis, dim_check)
  #   
  #   # Fixing basis type
  #   basis_choice = rep(basis_choice, dim_check)
  #   
  #   # Final update to domain range
  #   domain_range = domain_range_list
  #   
  #   # Warning
  #   print("Warning: You only specified basis information for one functional covariate -- it will be repeated for all functional covariates")
  #   
  # }
  # 
  # #### Creating functional observations in the case of raw data
  # if(raw_data == T){
  #   
  #   # Taking in data
  #   dat = func_cov
  #   
  #   # Setting up array
  #   temp_tensor = array(dim = c(31, nrow(dat[[1]]), length(dat)))
  #   
  #   for (t in 1:length(dat)) {
  #     
  #     # Getting appropriate obs
  #     curr_func = dat[[t]]
  #     
  #     # Getting current domain
  #     curr_domain = domain_range[[t]]
  #     
  #     # Creating basis (using bspline)
  #     basis_setup = create.bspline.basis(rangeval = c(curr_domain[1], curr_domain[2]),
  #                                        nbasis = 31,
  #                                        norder = 4)
  #     
  #     # Time points
  #     time_points = seq(curr_domain[1], curr_domain[2], length.out = ncol(curr_func))
  #     
  #     # Making functional observation
  #     temp_fd = Data2fd(time_points, t(curr_func), basis_setup)
  #     
  #     # Storing data
  #     temp_tensor[,,t] = temp_fd$coefs
  #     
  #   }
  #   
  #   # Saving as appropriate names
  #   func_cov = temp_tensor
  #   
  # }
  # 
  # ##### Helper Functions #####
  # 
  # # Composite approximator
  # composite_approximator <- function(f, a, b, n) {
  #   
  #   # This function does the integral approximations and gets called in the
  #   # integral approximator function. In the integral approximator function
  #   # we pass in a function f into this and that is final output - a collection
  #   # of numbers - one for each of the functional observations
  #   
  #   # Error checking code
  #   if (is.function(f) == FALSE) {
  #     stop('The input f(x) must be a function with one parameter (variable)')
  #   }
  #   
  #   # General formula
  #   h <- (b - a)/n
  #   
  #   # Setting parameters
  #   xn <- seq.int(a, b, length.out = n + 1)
  #   xn <- xn[-1]
  #   xn <- xn[-length(xn)]
  #   
  #   # Approximating using the composite rule formula
  #   integ_approx <- (h/3)*(f(a) + 2*sum(f(xn[seq.int(2, length(xn), 2)])) + 
  #                            4*sum(f(xn[seq.int(1, length(xn), 2)])) + 
  #                            f(b))
  #   
  #   # Returning result
  #   return(integ_approx)
  #   
  # }
  # 
  # # Integration Approximation for fourier and b-spline
  # integral_form_fourier <- function(functional_data, 
  #                                   beta_basis = NULL, 
  #                                   num_fd_basis = dim(func_cov)[1], 
  #                                   num_beta_basis,
  #                                   range){
  #   
  #   ########################################################################
  #   
  #   #### Setting up x_i(s) form ####
  #   
  #   # Initializing
  #   func_basis_sin <- c()
  #   func_basis_cos <- c()
  #   
  #   # Setting up vectors
  #   for (i in 1:((num_fd_basis - 1)/2)) {
  #     func_basis_sin[i] <- paste0("sin(2*pi*x*", i, "/", range[2], ")")
  #   }
  #   for (i in 1:((num_fd_basis - 1)/2)) {
  #     func_basis_cos[i] <- paste0("cos(2*pi*x*", i, "/", range[2], ")")
  #   }
  #   
  #   # Putting together
  #   fd_basis_form <- c(1, rbind(func_basis_sin, func_basis_cos))
  #   
  #   # Combining with functional data
  #   x_1s <- paste0(functional_data, "*", fd_basis_form, collapse = " + ")
  #   
  #   ########################################################################
  #   
  #   #### Setting up beta_(s) ####
  #   
  #   beta_basis_sin <- c()
  #   beta_basis_cos <- c()
  #   
  #   # Setting up vectors
  #   for (i in 1:((num_beta_basis - 1)/2)) {
  #     beta_basis_sin[i] <- paste0("sin(2*pi*x*", i, "/", range[2], ")")
  #   }
  #   for (i in 1:((num_beta_basis - 1)/2)) {
  #     beta_basis_cos[i] <- paste0("cos(2*pi*x*", i, "/", range[2], ")")
  #   }
  #   
  #   # Combining with functional data
  #   beta_basis_form <- c(1, rbind(beta_basis_sin, beta_basis_cos))
  #   
  #   ########################################################################
  #   
  #   #### Getting approximations ####
  #   
  #   # Initializing - should be vector of size 11
  #   integ_approximations <- c()
  #   
  #   for (i in 1:length(beta_basis_form)) {
  #     
  #     # Combining
  #     form_approximated <- paste0(beta_basis_form[i], "*(", x_1s, ")")
  #     
  #     # Passing to appropriate form
  #     final_func <- function(x){
  #       a = eval(parse(text = form_approximated))
  #       return(a)
  #     }
  #     
  #     # Evaluating
  #     integ_approximations[i] <- composite_approximator(final_func, range[1], range[2], 5000)
  #   }
  #   
  #   return(integ_approximations)
  #   
  # }
  # 
  # integral_form_bspline <- function(functional_data, 
  #                                   beta_basis = NULL, 
  #                                   num_fd_basis = dim(func_cov)[1], 
  #                                   num_beta_basis){
  #   
  # }
  # 
  # if(is.null(scalar_cov)){
  #   converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
  #                                     ncol = sum(num_basis)))
  # } else {
  #   converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
  #                                     ncol = sum(num_basis) + ncol(scalar_cov)))
  # }
  # 
  # # Looping to get approximations
  # for (i in 1:dim(func_cov)[3]) {
  #   
  #   # Current data set
  #   df <- func_cov[,,i]
  #   
  #   # Turning into matrix
  #   if(is.vector(df) == T){
  #     test_mat = matrix(nrow = length(df), ncol = 1)
  #     test_mat[,1] = df
  #     df = test_mat
  #   }
  #   
  #   # Current number of basis and choice of basis information
  #   cur_basis_num <- num_basis[i]
  #   cur_basis <- basis_choice[i]
  #   
  #   # Getting current range
  #   cur_range <- domain_range[[i]]
  #   
  #   # Storing previous numbers
  #   if(i == 1){
  #     left_end = 1
  #     right_end = cur_basis_num
  #   } else {
  #     left_end = sum(num_basis[1:(i - 1)]) + 1
  #     right_end = (left_end - 1) + cur_basis_num
  #   }
  #   
  #   if(cur_basis == "fourier"){
  #     for (j in 1:ncol(df)) {
  #       converted_df[j, left_end:right_end] <- c(integral_form_fourier(df[,j], 
  #                                                                      num_beta_basis = cur_basis_num,
  #                                                                      range = cur_range))
  #     }
  #   } else{
  #     
  #   }
  #   
  # }
  # 
  # # Now attaching scalar covariates
  # if(is.null(scalar_cov)){
  #   converted_df <- converted_df
  # } else{
  #   for (k in 1:nrow(converted_df)) {
  #     converted_df[k, (sum(num_basis) + 1):(sum(num_basis) + ncol(scalar_cov))] <- scalar_cov[k,]
  #   }
  # }
  # 
  # # Normalize training data
  # if(covariate_scaling == T){
  #   train_x <- scale(converted_df)
  # } else {
  #   train_x <- as.matrix(cbind(converted_df[,c(1:sum(num_basis))], scale(converted_df[,-c(1:sum(num_basis))])))
  # }
  
  # Returning the model
  return(list(data = train_x,
              raw_data = converted_df,
              fnc_basis_num = num_basis,
              fnc_type = basis_choice,
              func_obs = func_cov)
  )
}

# (1) FNN Function

FNN <- function(resp, 
                func_cov, 
                scalar_cov = NULL,
                basis_choice, 
                num_basis,
                hidden_layers,
                neurons_per_layer,
                activations_in_layers,
                domain_range,
                epochs,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                learn_rate = 0.001,
                patience_param = 15,
                early_stopping = T,
                print_info = T,
                batch_size = 32,
                decay_rate = 0){
  
  # # Required packages
  # library(keras)
  # library(tensorflow)
  # library(fda)
  # library(ggplot2)
  
  # DESCRIPTIONS OF PARAMETERS #
  
  # Resp = this is your response and will be dimension equal to the number of observations
  
  # func_cov = this will be a 3 dimensional tensor where the third dimension = K, the COLUMNS are the observations
  #            and the ROWS are the coefficient values (so the number of rows is the choice in number of basis functions)
  
  # scalar_cov = this is a matrix where the ROWS are the observations and the columns are the corresponding
  #              feature values
  
  # basis_choice = this will be updated more later but this should be a vector of names (e.g c("Fourier", "B-Spline")) of
  #                dimension K (the number of functional covariates)
  
  # num_basis = this will be the number of basis used for each of the K functional covariates. So, for example, this will
  #             be something like c(3, 7) meaning that there will be 3 basis functions defining the first functional
  #             covariate and 7 defining the second
  
  # hiden_layers = this is a single number equal to the number of hidden layers in the network (including the first layer)
  
  # neurons_per_layer = this will be a vector of size equal to the number of hidden layers
  
  # activations_in_layers = this is just the choice of activation function for each of the layers, same dimension as the
  #                         neurons_per_layer vector above
  
  # domain_range = list of size K of two digit numbers indicating the range for each of the functional covariates
  
  # Epochs = the number of times we go through the backward and forward pass (or rather, total run throughs of the data)
  
  # Output_size = the number of outputs you want at the end -> usually this is going to be 1 for regression
  
  # loss_choice = this is a keras parameter to see the loss criterion
  
  # metric_choice = this is another keras parameter
  
  # val_split = this is another keras parameter to split the training into a validation and actual training set
  
  # patience_param = this is another keras parameter used in early stopping
  
  # learn_rate = the learning rate for the optimizer
  
  # early_stop = this is a keras paramter; decide if you want to stop the training once there is not
  #              much improvement in the validation loss
  
  # END OF DESCRIPTIONS #
  
  ##### Helper Functions #####
  
  # Composite approximator
  composite_approximator <- function(f, a, b, n) {
    
    # This function does the integral approximations and gets called in the
    # integral approximator function. In the integral approximator function
    # we pass in a function f into this and that is final output - a collection
    # of numbers - one for each of the functional observations
    
    # Error checking code
    if (is.function(f) == FALSE) {
      stop('The input f(x) must be a function with one parameter (variable)')
    }
    
    # General formula
    h <- (b - a)/n
    
    # Setting parameters
    xn <- seq.int(a, b, length.out = n + 1)
    xn <- xn[-1]
    xn <- xn[-length(xn)]
    
    # Approximating using the composite rule formula
    integ_approx <- (h/3)*(f(a) + 2*sum(f(xn[seq.int(2, length(xn), 2)])) + 
                             4*sum(f(xn[seq.int(1, length(xn), 2)])) + 
                             f(b))
    
    # Returning result
    return(integ_approx)
    
  }
  
  # Integration Approximation for fourier and b-spline
  integral_form_fourier <- function(functional_data, 
                                 beta_basis = NULL, 
                                 num_fd_basis = dim(func_cov)[1], 
                                 num_beta_basis,
                                 range){
    
    ########################################################################
    
    #### Setting up x_i(s) form ####
    
    # Initializing
    func_basis_sin <- c()
    func_basis_cos <- c()
    
    # Setting up vectors
    for (i in 1:((num_fd_basis - 1)/2)) {
      func_basis_sin[i] <- paste0("sin(2*pi*x*", i, "/", range[2], ")")
    }
    for (i in 1:((num_fd_basis - 1)/2)) {
      func_basis_cos[i] <- paste0("cos(2*pi*x*", i, "/", range[2], ")")
    }
    
    # Putting together
    fd_basis_form <- c(1, rbind(func_basis_sin, func_basis_cos))
    
    # Combining with functional data
    x_1s <- paste0(functional_data, "*", fd_basis_form, collapse = " + ")
    
    ########################################################################
    
    #### Setting up beta_(s) ####
    
    beta_basis_sin <- c()
    beta_basis_cos <- c()
    
    # Setting up vectors
    for (i in 1:((num_beta_basis - 1)/2)) {
      beta_basis_sin[i] <- paste0("sin(2*pi*x*", i, "/", range[2], ")")
    }
    for (i in 1:((num_beta_basis - 1)/2)) {
      beta_basis_cos[i] <- paste0("cos(2*pi*x*", i, "/", range[2], ")")
    }
    
    # Combining with functional data
    beta_basis_form <- c(1, rbind(beta_basis_sin, beta_basis_cos))
    
    ########################################################################
    
    #### Getting approximations ####
    
    # Initializing - should be vector of size 11
    integ_approximations <- c()
    
    for (i in 1:length(beta_basis_form)) {
      
      # Combining
      form_approximated <- paste0(beta_basis_form[i], "*(", x_1s, ")")
      
      # Passing to appropriate form
      final_func <- function(x){
        a = eval(parse(text = form_approximated))
        return(a)
      }
      
      # Evaluating
      integ_approximations[i] <- composite_approximator(final_func, range[1], range[2], 5000)
    }
    
    return(integ_approximations)
    
  }
  
  integral_form_bspline <- function(functional_data, 
                                    beta_basis = NULL, 
                                    num_fd_basis = dim(func_cov)[1], 
                                    num_beta_basis){
    
  }
  
  
  # First, we need to create the proper data set. This means to get the approximations and append
  # them together for each of the covariates. We are asking for the user to pass an array where the
  # third dimension is equal to K = the number of functional covariates. Each of these will contain
  # the coefficients as found by turning the data into a functional data object.
  
  # Initializing matrix to keep everything inside across all functional covariates
  
  ######### TEST CODE ############
  #scalar_cov = NULL
  ################################
  
  if(is.null(scalar_cov)){
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis)))
  } else {
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis) + ncol(scalar_cov)))
  }
  
  ######### TEST CODE ############
  #func_cov = test_data
  #num_basis = c(3, 5)
  #basis_choice = c("fourier", "fourier")
  #domain_range = list(c(1, 24), c(1, 24))
  ################################
  
  # Looping to get approximations
  for (i in 1:dim(func_cov)[3]) {
    
    # Current data set
    df <- func_cov[,,i]
    
    # Turning into matrix
    if(is.vector(df) == T){
      test_mat = matrix(nrow = length(df), ncol = 1)
      test_mat[,1] = df
      df = test_mat
    }
    
    # Current number of basis and choice of basis information
    cur_basis_num <- num_basis[i]
    cur_basis <- basis_choice[i]
    
    # Getting current range
    cur_range <- domain_range[[i]]
    
    # Storing previous numbers
    if(i == 1){
      left_end = 1
      right_end = cur_basis_num
    } else {
      left_end = sum(num_basis[1:(i - 1)]) + 1
      right_end = (left_end - 1) + cur_basis_num
    }
    
    if(cur_basis == "fourier"){
      for (j in 1:ncol(df)) {
        converted_df[j, left_end:right_end] <- c(integral_form_fourier(df[,j], 
                                                       num_beta_basis = cur_basis_num,
                                                       range = cur_range))
      }
    } else{
      
    }
    
  }
  
  # Now attaching scalar covariates
  if(is.null(scalar_cov)){
    converted_df <- converted_df
  } else{
    for (k in 1:nrow(converted_df)) {
      converted_df[k, (sum(num_basis) + 1):(sum(num_basis) + ncol(scalar_cov))] <- scalar_cov[k,]
    }
  }
  
  # Now we have the data set to pass onto the network, we can set up the data so that it is well suited to be
  # passed onto the network. This means normalizing things and rewriting some other things
  
  # Normalize training data
  train_x <- scale(converted_df) 
  train_y <- resp
  
  # Now, we can move onto creating the model. This means taking advantage of the last three variables. We will use another
  # function to do this that lets us add layers easily.
  
  ######### TEST CODE ############
  #hidden_layers = 4
  #activations_in_layers = c("relu", "relu", "relu", "relu")
  #neurons_per_layer = c(64, 64, 64, 64)
  #output_size = 1
  #loss_choice = "mse"
  #metric_choice = list("mean_squared_error")
  ################################
  
  # Creating model
  build_model <- function(train_x,
                          neurons_per_layer, 
                          activations_in_layers,
                          hidden_layers,
                          output_size,
                          loss_choice,
                          metric_choice) {
    
    # Initializing model for FNN layer
    model <- keras_model_sequential() %>%
      layer_dense(units = neurons_per_layer[1], activation = activations_in_layers[1],
                  input_shape = dim(train_x)[2])
    
    # Adding in additional model layers
    if(hidden_layers > 1){
      for (i in 1:(hidden_layers - 1)) {
        model <- model %>% layer_dense(units = neurons_per_layer[i + 1], activation = activations_in_layers[i + 1])
      }
    }
    
    # Setting up final layer
    model <- model %>% layer_dense(units = output_size)
    
    # Setting up other model parameters
    model %>% compile(
      loss = loss_choice,
      optimizer = optimizer_adam(lr = learn_rate, decay = decay_rate),
      metrics = metric_choice
    )
    
    return(model)
  }
  
  # Now we have the model set up, we can begin to initialize the network before it is ultimately trained. This will also
  # print out a summary of the model thus far
  model <- build_model(train_x,
                       neurons_per_layer, 
                       activations_in_layers,
                       hidden_layers,
                       output_size,
                       loss_choice,
                       metric_choice)
  
  if(print_info ==  T){
    print(model)  
  }
  
  # We can also display the progress of the network to make it easier to visualize using the following. This is
  # borrowed from the keras write up for R on the official website
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 80 == 0) cat("\n")
      cat("x")
    }
  )  
  
  # The patience parameter is the amount of epochs to check for improvement.
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience_param)
  
  # Now finally, we can fit the model
  if(early_stopping == T & print_info == T){
    history <- model %>% fit(
      train_x,
      train_y,
      epochs = epochs,
      batch_size = batch_size,
      validation_split = val_split,
      verbose = 0,
      callbacks = list(early_stop, print_dot_callback)
    )
  } else if(early_stopping == T & print_info == F) {
    history <- model %>% fit(
      train_x,
      train_y,
      epochs = epochs,
      validation_split = val_split,
      verbose = 0,
      callbacks = list(early_stop)
    )
  } else if(early_stopping == F & print_info == T){
    history <- model %>% fit(
      train_x,
      train_y,
      epochs = epochs,
      validation_split = val_split,
      verbose = 0,
      callbacks = list(print_dot_callback)
    )
  } else {
    history <- model %>% fit(
      train_x,
      train_y,
      epochs = epochs,
      validation_split = val_split,
      verbose = 0,
      callbacks = list()
    )
  }
  
  
  # Plotting the errors
  if(print_info == T){
    print(plot(history, metrics = "mean_squared_error", smooth = FALSE) + 
            theme_bw() + 
            xlab("Epoch Number") + 
            ylab(""))
  }
  
  # Skipping line
  cat("\n")
  
  # Printing out
  if(print_info == T){
    print(history)
  }
  
  # Returning the model
  return(list(model = model, 
              data = train_x, 
              fnc_basis_num = num_basis,
              fnc_type = basis_choice,
              parameter_info = history$params,
              per_iter_info = history$metrics))
}

# (2) Predict Function

FNN_Predict = function(model,
                       func_cov, 
                       scalar_cov = NULL,
                       basis_choice, 
                       num_basis,
                       domain_range){
  
  ##### Helper Functions #####
  
  # Composite approximator
  composite_approximator <- function(f, a, b, n) {
    
    # This function does the integral approximations and gets called in the
    # integral approximator function. In the integral approximator function
    # we pass in a function f into this and that is final output - a collection
    # of numbers - one for each of the functional observations
    
    # Error checking code
    if (is.function(f) == FALSE) {
      stop('The input f(x) must be a function with one parameter (variable)')
    }
    
    # General formula
    h <- (b - a)/n
    
    # Setting parameters
    xn <- seq.int(a, b, length.out = n + 1)
    xn <- xn[-1]
    xn <- xn[-length(xn)]
    
    # Approximating using the composite rule formula
    integ_approx <- (h/3)*(f(a) + 2*sum(f(xn[seq.int(2, length(xn), 2)])) + 
                             4*sum(f(xn[seq.int(1, length(xn), 2)])) + 
                             f(b))
    
    # Returning result
    return(integ_approx)
    
  }
  
  # Integration Approximation for fourier and b-spline
  integral_form_fourier <- function(functional_data, 
                                    beta_basis = NULL, 
                                    num_fd_basis = dim(func_cov)[1], 
                                    num_beta_basis,
                                    range){
    
    ########################################################################
    
    #### Setting up x_i(s) form ####
    
    # Initializing
    func_basis_sin <- c()
    func_basis_cos <- c()
    
    # Setting up vectors
    for (i in 1:((num_fd_basis - 1)/2)) {
      func_basis_sin[i] <- paste0("sin(2*pi*x*", i, "/", range[2], ")")
    }
    for (i in 1:((num_fd_basis - 1)/2)) {
      func_basis_cos[i] <- paste0("cos(2*pi*x*", i, "/", range[2], ")")
    }
    
    # Putting together
    fd_basis_form <- c(1, rbind(func_basis_sin, func_basis_cos))
    
    # Combining with functional data
    x_1s <- paste0(functional_data, "*", fd_basis_form, collapse = " + ")
    
    ########################################################################
    
    #### Setting up beta_(s) ####
    
    beta_basis_sin <- c()
    beta_basis_cos <- c()
    
    # Setting up vectors
    for (i in 1:((num_beta_basis - 1)/2)) {
      beta_basis_sin[i] <- paste0("sin(2*pi*x*", i, "/", range[2], ")")
    }
    for (i in 1:((num_beta_basis - 1)/2)) {
      beta_basis_cos[i] <- paste0("cos(2*pi*x*", i, "/", range[2], ")")
    }
    
    # Combining with functional data
    beta_basis_form <- c(1, rbind(beta_basis_sin, beta_basis_cos))
    
    ########################################################################
    
    #### Getting approximations ####
    
    # Initializing - should be vector of size whatever
    integ_approximations <- c()
    
    for (i in 1:length(beta_basis_form)) {
      
      # Combining
      form_approximated <- paste0(beta_basis_form[i], "*(", x_1s, ")")
      
      # Passing to appropriate form
      final_func <- function(x){
        a = eval(parse(text = form_approximated))
        return(a)
      }
      
      # Evaluating
      integ_approximations[i] <- composite_approximator(final_func, range[1], range[2], 5000)
    }
    
    return(integ_approximations)
    
  }
  
  integral_form_bspline <- function(functional_data, 
                                    beta_basis = NULL, 
                                    num_fd_basis = dim(func_cov)[1], 
                                    num_beta_basis){
    
  }
  
  # First, we need to create the proper data set. This means to get the approximations and append
  # them together for each of the covariates. We are asking for the user to pass an array where the
  # third dimension is equal to K = the number of functional covariates. Each of these will contain
  # the coefficients as found by turning the data into a functional data object.
  
  # Initializing matrix to keep everything inside across all functional covariates
  
  ######### TEST CODE ############
  #scalar_cov = NULL
  ################################
  
  if(is.null(scalar_cov)){
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis)))
  } else {
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis) + ncol(scalar_cov)))
  }
  
  ######### TEST CODE ############
  #func_cov = bike_data_test
  #num_basis = c(3, 5)
  #basis_choice = c("fourier", "fourier")
  #domain_range = list(c(1, 24), c(1, 24))
  ################################
  
  # Looping to get approximations
  for (i in 1:dim(func_cov)[3]) {
    
    # Current data set
    df <- func_cov[,,i]
    
    # Turning into matrix
    if(is.vector(df) == T){
      test_mat = matrix(nrow = length(df), ncol = 1)
      test_mat[,1] = df
      df = test_mat
    }
    
    # Current number of basis and choice of basis information
    cur_basis_num <- num_basis[i]
    cur_basis <- basis_choice[i]
    
    # Getting current range
    cur_range <- domain_range[[i]]
    
    # Storing previous numbers
    if(i == 1){
      left_end = 1
      right_end = cur_basis_num
    } else {
      left_end = sum(num_basis[1:(i - 1)]) + 1
      right_end = (left_end - 1) + cur_basis_num
    }
    
    if(cur_basis == "fourier"){
      for (j in 1:ncol(df)) {
        converted_df[j, left_end:right_end] <- c(integral_form_fourier(df[,j], 
                                                                       num_beta_basis = cur_basis_num,
                                                                       range = cur_range))
      }
    } else{
      
    }
    
  }
  
  # Now attaching scalar covariates
  if(is.null(scalar_cov)){
    converted_df <- converted_df
  } else{
    for (k in 1:nrow(converted_df)) {
      converted_df[k, (sum(num_basis) + 1):(sum(num_basis) + ncol(scalar_cov))] <- scalar_cov[k,]
    }
  }
  
  
  # Now we have the data set to pass onto the network, we can set up the data so that it is well suited to be
  # passed onto the network. This means normalizing things and rewriting some other things
  
  # Use means and standard deviations from training set to normalize test set
  
  ######### TEST CODE ############
  #model = bike_example
  ################################
  
  col_means_train <- attr(model$data, "scaled:center") 
  col_stddevs_train <- attr(model$data, "scaled:scale")
  test_x <- scale(converted_df, center = col_means_train, scale = col_stddevs_train)
  
  # Predicting
  test_predictions <- model$model %>% predict(test_x)
  
  # Returning prediction 
  return(prediction = test_predictions[ , 1])
  
}


# (3) Functionl Neural Coefficient Function

FNN_FNC = function(model, domain_range){
  
  # Libraries
  #library(ggpubr)
  
  # Getting weights
  weights = rowMeans(get_weights(model$model)[[1]])
  
  # Creating list to separate
  fnc_list = list()
  
  # Separating out for multiple FNCs
  for (i in 1:length(model$fnc_basis_num)) {
    
    # Current number of basis and choice of basis information
    cur_basis_num <- model$fnc_basis_num[i]
    
    # Storing previous numbers
    if(i == 1){
      left_end = 1
      right_end = cur_basis_num
    } else {
      left_end = sum(model$fnc_basis_num[1:(i - 1)]) + 1
      right_end = (left_end - 1) + cur_basis_num
    }
    
    fnc_list[[i]] = weights[left_end:right_end]
  }
  
  # Creating function - handles up to 51 basis term coefficient weight functions
  final_beta_fnn_fourier <- function(x, d, range){
    
    # Appending on 0s
    zero_vals = rep(0, 51 - length(d))
    
    # creating c vector
    c = c(d, zero_vals)
    
    # Getting values
    value <- c[1] + c[2]*sin(1*2*pi*x/range[2]) + c[3]*cos(1*2*pi*x/range[2]) +
      c[4]*sin(2*2*pi*x/range[2]) + c[5]*cos(2*2*pi*x/range[2]) + c[6]*sin(3*2*pi*x/range[2]) + c[7]*cos(3*2*pi*x/range[2]) +
      c[8]*sin(4*2*pi*x/range[2]) + c[9]*cos(4*2*pi*x/range[2]) + c[10]*sin(5*2*pi*x/range[2]) + c[11]*cos(5*2*pi*x/range[2]) +
      c[12]*sin(6*2*pi*x/range[2]) + c[13]*cos(6*2*pi*x/range[2]) + c[14]*sin(7*2*pi*x/range[2]) + c[15]*cos(7*2*pi*x/range[2]) +
      c[16]*sin(8*2*pi*x/range[2]) + c[17]*cos(8*2*pi*x/range[2]) + c[18]*sin(9*2*pi*x/range[2]) + c[19]*cos(9*2*pi*x/range[2]) +
      c[20]*sin(10*2*pi*x/range[2]) + c[21]*cos(10*2*pi*x/range[2]) + c[22]*sin(11*2*pi*x/range[2]) + c[23]*cos(11*2*pi*x/range[2]) +
      c[24]*sin(12*2*pi*x/range[2]) + c[25]*cos(12*2*pi*x/range[2]) + c[26]*sin(13*2*pi*x/range[2]) + c[27]*cos(13*2*pi*x/range[2]) +
      c[28]*sin(14*2*pi*x/range[2]) + c[29]*cos(14*2*pi*x/range[2]) + c[30]*sin(15*2*pi*x/range[2]) + c[31]*cos(15*2*pi*x/range[2]) +
      c[32]*sin(16*2*pi*x/range[2]) + c[33]*cos(16*2*pi*x/range[2]) + c[34]*sin(17*2*pi*x/range[2]) + c[35]*cos(17*2*pi*x/range[2]) +
      c[36]*sin(18*2*pi*x/range[2]) + c[37]*cos(18*2*pi*x/range[2]) + c[38]*sin(19*2*pi*x/range[2]) + c[39]*cos(19*2*pi*x/range[2]) +
      c[40]*sin(20*2*pi*x/range[2]) + c[41]*cos(20*2*pi*x/range[2]) + c[42]*sin(21*2*pi*x/range[2]) + c[43]*cos(21*2*pi*x/range[2]) +
      c[44]*sin(22*2*pi*x/range[2]) + c[45]*cos(22*2*pi*x/range[2]) + c[46]*sin(23*2*pi*x/range[2]) + c[47]*cos(23*2*pi*x/range[2]) +
      c[48]*sin(24*2*pi*x/range[2]) + c[49]*cos(24*2*pi*x/range[2]) + c[50]*sin(25*2*pi*x/range[2]) + c[51]*cos(25*2*pi*x/range[2])
    
    # Returning
    return(value)
    
  }
  
  final_beta_fnn_bspline <- function(x, d, range){
    
  }
  
  # Initializing plot list
  plots_saved = list()
  
  # Looping to get FNCs
  for (j in 1:length(fnc_list)) {
    
    # Current basis type
    current_basis = model$fnc_type[j]
    
    # Current range
    current_range = domain_range[[j]]
    
    if(current_basis == "fourier"){
      
      # Evaluating
      vals = final_beta_fnn_fourier(seq(current_range[1], current_range[2], 0.5), fnc_list[[j]], range = current_range)
      
    } else {
      
    }
    
    
    # Getting updated function
    beta_coef_fnn <- data.frame(continuum = seq(current_range[1], current_range[2], 0.5), 
                                beta_evals = vals)
    
    # ggplot return
    plots_saved[[j]] = beta_coef_fnn %>% 
                        ggplot(aes(x = continuum, y = beta_evals)) +
                        geom_line(size = 1.5, color = "blue") +
                        theme_bw() +
                        xlab("Continuum") +
                        ylab("beta(s)") +
                        ggtitle(paste0("Functional Neural Coefficient: ", j))
                        theme(plot.title = element_text(hjust = 0.5)) +
                        theme(axis.text=element_text(size=12),
                              axis.title=element_text(size=14,face="bold"))
  }
  
  # Printing Plots
  print(ggarrange(plotlist = plots_saved, ncol = length(fnc_list), nrow = 1))
  
  # Returning information
  return(list(FNC_Coefficients = fnc_list,
              saved_plot = plots_saved))
}


# (4) Cross-Validation Function

FNN_CV <- function(nfolds,
                  resp, 
                  func_cov, 
                  scalar_cov = NULL,
                  basis_choice, 
                  num_basis,
                  hidden_layers,
                  neurons_per_layer,
                  activations_in_layers,
                  domain_range,
                  epochs,
                  output_size = 1,
                  loss_choice = "mse",
                  metric_choice = list("mean_squared_error"),
                  val_split = 0.2,
                  learn_rate = 0.001,
                  patience_param = 15,
                  early_stopping = T,
                  print_info = T,
                  batch_size = 32,
                  decay_rate = 0){
  
  # Creating folds
  ###### TEST CODE #######
  #resp = annualprec
  #nfolds = 35
  ########################
  folds <- createFolds(resp, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  # Predictions initialization
  predictions = list()
  true_values = list()
  MSPE_Fold = list()
  differences = c()
  
  # Looping to create models
  for (i in 1:nfolds) {
    
    ####### TEST CODE ########
    # func_cov = temp_data
    # i = 1
    ##########################
    
    if(dim(func_cov)[3] == 1){
      
      # Initializing arrays
      data_train <- array(data = NA, dim = c(dim(func_cov)[1], 
                                             do.call(sum, lapply(folds, length)) - length(folds[[i]]),
                                             dim(func_cov)[3]))
      
      data_test <- array(data = NA, dim = c(dim(func_cov)[1], 
                                            length(folds[[i]]),
                                            dim(func_cov)[3]))
      
      # Splitting into test and train
      data_train[,,1] = func_cov[, -folds[[i]], ]
      data_test[,,1] = func_cov[, folds[[i]], ]
      resp_train = resp[-folds[[i]]]
      resp_test = resp[folds[[i]]]
      
    } else if (dim(func_cov)[2] == nfolds) {
      
      # Initializing
      data_test = array(data = NA, dim = c(dim(func_cov)[1], 
                                           1,
                                           dim(func_cov)[3]))
      
      # Splitting into test and train
      data_train = func_cov[, -folds[[i]], ]
      data_test[,1,] = func_cov[, folds[[i]], ]
      resp_train = resp[-folds[[i]]]
      resp_test = resp[folds[[i]]]
    } else {
      
      # Splitting into test and train
      data_train = func_cov[, -folds[[i]], ]
      data_test = func_cov[, folds[[i]], ]
      resp_train = resp[-folds[[i]]]
      resp_test = resp[folds[[i]]]
    }
    
    # Saving
    true_values[[i]] = resp_test
    
    ##### TESTING THINGS ###############################################################
    # basis_choice = c("fourier")
    # domain_range = list(c(1, 24))
    # scalar_cov = NULL
    # num_basis = as.numeric(as.character((x[(current_layer + 1):(length(basis_choice) + current_layer)])))
    # hidden_layers = current_layer
    # neurons_per_layer = as.numeric(as.character(x[(length(basis_choice) + current_layer + 1):((length(basis_choice) + current_layer) + current_layer)]))
    # activations_in_layers = as.character(x[1:current_layer])
    # epochs = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 1]))
    # output_size = 1
    # loss_choice = "mse"
    # metric_choice = list("mean_squared_error")
    # val_split = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 2]))
    # patience_param = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 3]))
    # early_stopping = T
    # print_info = F
    ####################################################################################
    # scalar_cov = NULL
    # basis_choice = c("fourier")
    # num_basis = 11
    # hidden_layers = 3
    # neurons_per_layer = c(128, 64, 64)
    # activations_in_layers = c("sigmoid", "sigmoid", "linear")
    # domain_range = list(c(1, 365))
    # epochs = 200
    # output_size = 1
    # loss_choice = "mse"
    # metric_choice = list("mean_squared_error")
    # val_split = 0.15
    # learn_rate = 0.0001
    # patience_param = 25
    # early_stopping = F
    # print_info = T
    ####################################################################################
    
    # Running model
    model_cv = FNN(resp_train, 
                   data_train, 
                   scalar_cov,
                   basis_choice, 
                   num_basis,
                   hidden_layers,
                   neurons_per_layer,
                   activations_in_layers,
                   domain_range,
                   epochs,
                   output_size,
                   loss_choice,
                   metric_choice,
                   val_split,
                   learn_rate,
                   patience_param,
                   early_stopping,
                   print_info,
                   batch_size,
                   decay_rate)
    
    # Predicting
    predictions[[i]] = FNN_Predict(model_cv,
                                   data_test, 
                                   scalar_cov,
                                   basis_choice, 
                                   num_basis,
                                   domain_range)
    
    # Getting differences
    differences = c(differences, predictions[[i]] - resp_test)
    
    # Getting MSPE per fold
    MSPE_Fold[[i]] = mean((predictions[[i]] - true_values[[i]])^2)

    # Folds done
    if(print_info == T){
      cat("\n")
      print(paste0("Folds Done: ", i))
    }
    
  }
  
  # Calculating MSPE
  MSPE_to_return = mean(differences^2)
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
  
  # Returning
  return(list(predicted_folds = predictions,
              true_folds = true_values,
              MSPE = list(Overall_MSPE = MSPE_to_return,
                          Fold_MSPEs = MSPE_Fold),
              fold_indices = folds))
  
}

# (5) Tuning Function

FNN_Tune = function(tune_list,
                    resp, 
                    func_cov,
                    basis_choice,
                    domain_range,
                    batch_size = 32,
                    decay_rate = 0,
                    nfolds = 5,
                    cores = 4){
  
  # Parallel apply set up
  #plan(multiprocess, workers = cores)
  
  # Setting up function
  tune_func = function(x,
                       nfolds,
                       resp, 
                       func_cov,
                       basis_choice,
                       domain_range,
                       batch_size,
                       decay_rate){
    
    # Setting seed
    use_session_with_seed(
      1,
      disable_gpu = F,
      disable_parallel_cpu = F,
      quiet = T
    )
    
    # Clearing irrelevant information
    colnames(x) <- NULL
    rownames(x) <- NULL
    
    # Running model
    model_results = FNN_CV(nfolds,
                           resp, 
                           func_cov = func_cov, 
                           scalar_cov = NULL,
                           basis_choice = basis_choice, 
                           num_basis = as.numeric(as.character((x[(current_layer + 1):(length(basis_choice) + current_layer)]))),
                           hidden_layers = current_layer,
                           neurons_per_layer = as.numeric(as.character(x[(length(basis_choice) + current_layer + 1):((length(basis_choice) + current_layer) + current_layer)])),
                           activations_in_layers = as.character(x[1:current_layer]),
                           domain_range = domain_range,
                           epochs = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 1])),
                           output_size = 1,
                           loss_choice = "mse",
                           metric_choice = list("mean_squared_error"),
                           val_split = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 2])),
                           learn_rate = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 4])),
                           patience_param = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 3])),
                           early_stopping = T,
                           print_info = F,
                           batch_size = batch_size,
                           decay_rate = decay_rate)
    
    # Putting together
    list_returned <- list(MSPE = model_results$MSPE$Overall_MSPE,
                          num_basis = as.numeric(as.character((x[(current_layer + 1):(length(basis_choice) + current_layer)]))),
                          hidden_layers = current_layer,
                          neurons_per_layer = as.numeric(as.character(x[(length(basis_choice) + current_layer + 1):((length(basis_choice) + current_layer) + current_layer)])),
                          activations_in_layers = as.character(x[1:current_layer]),
                          epochs = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 1])),
                          val_split = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 2])),
                          patience_param = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 3])),
                          learn_rate = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 4])))
    
    # Clearing backend
    K <- backend()
    K$clear_session()
    
    # Returning
    return(list_returned)
    
  }
  
  # Saving MSPEs
  Errors = list()
  All_Errors = list()
  Grid_List = list()
  
  # Setting up tuning parameters
  for (i in 1:length(tune_list$num_hidden_layers)) {
    
    ###### TEST CODE ############
    # i = 2
    # tune_list = tune_list_bike
    # basis_choice = "fourier"
    # tune_list_bike
    # resp = rentals
    # func_cov = bike_data_tune
    # 1:dim(func_cov)[3]
    # basis_choice = c("fourier")
    # domain_range = list(c(1, 24))
    # nfolds = 2
    # cores = 4
    # learn_rate = 0.0001
    #############################
    
    # Current layer number
    current_layer = tune_list$num_hidden_layers[i]
    
    # Creating data frame of list
    df = expand.grid(rep(list(tune_list$neurons), tune_list$num_hidden_layers[i]), stringsAsFactors = F)
    df2 = expand.grid(rep(list(tune_list$num_basis), length(basis_choice)), stringsAsFactors = F)
    df3 = expand.grid(rep(list(tune_list$activation_choice), tune_list$num_hidden_layers[i]), stringsAsFactors = F)
    colnames(df2)[length(basis_choice)] <- "Var2.y"
    colnames(df3)[i] <- "Var2.z"
    
    # Getting grid
    pre_grid = expand.grid(df$Var1,
                           Var2.y = df2$Var2.y,
                           Var2.z = df3$Var2.z,
                           tune_list$epochs,
                           tune_list$val_split,
                           tune_list$patience,
                           tune_list$learn_rate)
    
    # Merging
    combined <- unique(merge(df, pre_grid, by = "Var1"))
    combined2 <- unique(merge(df2, combined, by = "Var2.y"))
    final_grid <- suppressWarnings(unique(merge(df3, combined2, by = "Var2.z"))) 
    
    # Saving grid
    Grid_List[[i]] = final_grid
      
    # Now, we can pass on the combinations to the model
    results = pbapply(final_grid, 1, tune_func, 
                             nfolds = nfolds,
                             resp = resp, 
                             func_cov = func_cov,
                             basis_choice = basis_choice,
                             domain_range = domain_range,
                             batch_size = batch_size,
                             decay_rate = decay_rate)
    
    # Initializing
    MSPE_vals = c()
    
    # Collecting results
    for (u in 1:length(results)) {
      MSPE_vals[u] <- as.vector(results[[u]][1])
    }
    
    # All Errors
    All_Errors[[i]] = results
    
    # Getting best
    Errors[[i]] = results[[which.min(do.call(c, MSPE_vals))]]
      
    # Printing where we are at
    cat("\n")
    print(paste0("Done tuning for: ", current_layer, " hidden layers."))
  
  }
  
  # Initializing
  MSPE_after = c()
  
  # Getting best set of parameters
  for (i in 1:length(tune_list$num_hidden_layers)) {
    MSPE_after[i] = Errors[[i]]$MSPE
  }
  
  # Selecting minimum
  best = which.min(MSPE_after)

  # Returning best set of parameters
  return(list(Parameters = Errors[[best]],
              All_Information = All_Errors,
              Best_Per_Layer = Errors,
              Grid_List = Grid_List))
  
}


##### Helper Functions #####

# Rescale Function
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

# Range Function
range_01 <- function(x){(x-min(x))/(max(x)-min(x))}

# R squared function
rsq <- function(x, y){
  return(cor(x, y)^2)
} 

# Getting functional coef (weather)
beta_lm_weather <- function(x, c){
  value <- c[1] + c[2]*sin(1*2*pi*x/365) + c[3]*cos(1*2*pi*x/365) +
    c[4]*sin(2*2*pi*x/365) + c[5]*cos(2*2*pi*x/365) + c[6]*sin(3*2*pi*x/365) +
    c[7]*cos(3*2*pi*x/365) + c[8]*sin(4*2*pi*x/365) + c[9]*cos(4*2*pi*x/365) +
    c[10]*sin(5*2*pi*x/365) + c[11]*cos(5*2*pi*x/365) + c[12]*sin(6*2*pi*x/365) +
    c[13]*cos(5*2*pi*x/365)
  return(value)
}

beta_fnn_weather_eg <- function(x, c){
  value <- c[1] + c[2]*sin(1*2*pi*x/365) + c[3]*cos(1*2*pi*x/365) +
    c[4]*sin(2*2*pi*x/365) + c[5]*cos(2*2*pi*x/365) + c[6]*sin(3*2*pi*x/365) +
    c[7]*cos(3*2*pi*x/365) + c[8]*sin(4*2*pi*x/365) + c[9]*cos(4*2*pi*x/365) +
    c[10]*sin(5*2*pi*x/365) + c[11]*cos(5*2*pi*x/365) + c[12]*sin(6*2*pi*x/365) +
    c[13]*cos(5*2*pi*x/365)
  return(value)
}

beta_fnn_weather <- function(x, c){
  value <- c[1] + c[2]*sin(1*2*pi*x/365) + c[3]*cos(1*2*pi*x/365) +
    c[4]*sin(2*2*pi*x/365) + c[5]*cos(2*2*pi*x/365) 
  return(value)
}

# Getting functional coef (bike)
beta_lm_bike <- function(x, c){
  value <- c[1] + c[2]*sin(1*2*pi*x/24) + c[3]*cos(1*2*pi*x/24) +
    c[4]*sin(2*2*pi*x/24) + c[5]*cos(2*2*pi*x/24) + c[6]*sin(3*2*pi*x/24) +
    c[7]*cos(3*2*pi*x/24) + c[8]*sin(4*2*pi*x/24) + c[9]*cos(4*2*pi*x/24) +
    c[10]*sin(5*2*pi*x/24) + c[11]*cos(5*2*pi*x/24)
  return(value)
}

beta_fnn_bike <- function(x, c){
  value <- c[1] + c[2]*sin(1*2*pi*x/24) + c[3]*cos(1*2*pi*x/24) 
    c[4]*sin(2*2*pi*x/24) + c[5]*cos(2*2*pi*x/24) + c[6]*sin(3*2*pi*x/24) +
    c[7]*cos(3*2*pi*x/24) + c[8]*sin(4*2*pi*x/24) + c[9]*cos(4*2*pi*x/24) +
    c[10]*sin(5*2*pi*x/24) + c[11]*cos(5*2*pi*x/24) + c[12]*sin(6*2*pi*x/24) + c[13]*cos(6*2*pi*x/24) +
      c[14]*sin(7*2*pi*x/24) + c[15]*cos(7*2*pi*x/24) 
  return(value)
}

# Composite approximator other
composite_approximator_other <- function(f, a, b, n, x_obs, beta) {
  
  # This function does the integral approximations and gets called in the
  # integral approximator function. In the integral approximator function
  # we pass in a function f into this and that is final output - a collection
  # of numbers - one for each of the functional observations
  
  # Error checking code
  if (is.function(f) == FALSE) {
    stop('The input f(x) must be a function with one parameter (variable)')
  }
  
  # General formula
  h <- (b - a)/n
  
  # Setting parameters
  xn <- seq.int(a, b, length.out = n + 1)
  xn <- xn[-1]
  xn <- xn[-length(xn)]
  
  # Approximating using the composite rule formula
  integ_approx <- (h/3)*(f(a, x_obs, beta) + 2*sum(f(xn[seq.int(2, length(xn), 2)], x_obs, 
                                                     beta)) + 
                           4*sum(f(xn[seq.int(1, length(xn), 2)],
                                   x_obs, beta)) + 
                           f(b, x_obs, beta))
  
  # Returning result
  return(integ_approx)
  
}

# Creating function - handles up to 51 basis term coefficient weight functions
final_beta_fourier <- function(x, d, range){
  
  # Appending on 0s
  zero_vals = rep(0, 51 - length(d))
  
  # creating c vector
  c = c(d, zero_vals)
  
  # Getting values
  value <- c[1] + c[2]*sin(1*2*pi*x/range[2]) + c[3]*cos(1*2*pi*x/range[2]) +
    c[4]*sin(2*2*pi*x/range[2]) + c[5]*cos(2*2*pi*x/range[2]) + c[6]*sin(3*2*pi*x/range[2]) + c[7]*cos(3*2*pi*x/range[2]) +
    c[8]*sin(4*2*pi*x/range[2]) + c[9]*cos(4*2*pi*x/range[2]) + c[10]*sin(5*2*pi*x/range[2]) + c[11]*cos(5*2*pi*x/range[2]) +
    c[12]*sin(6*2*pi*x/range[2]) + c[13]*cos(6*2*pi*x/range[2]) + c[14]*sin(7*2*pi*x/range[2]) + c[15]*cos(7*2*pi*x/range[2]) +
    c[16]*sin(8*2*pi*x/range[2]) + c[17]*cos(8*2*pi*x/range[2]) + c[18]*sin(9*2*pi*x/range[2]) + c[19]*cos(9*2*pi*x/range[2]) +
    c[20]*sin(10*2*pi*x/range[2]) + c[21]*cos(10*2*pi*x/range[2]) + c[22]*sin(11*2*pi*x/range[2]) + c[23]*cos(11*2*pi*x/range[2]) +
    c[24]*sin(12*2*pi*x/range[2]) + c[25]*cos(12*2*pi*x/range[2]) + c[26]*sin(13*2*pi*x/range[2]) + c[27]*cos(13*2*pi*x/range[2]) +
    c[28]*sin(14*2*pi*x/range[2]) + c[29]*cos(14*2*pi*x/range[2]) + c[30]*sin(15*2*pi*x/range[2]) + c[31]*cos(15*2*pi*x/range[2]) +
    c[32]*sin(16*2*pi*x/range[2]) + c[33]*cos(16*2*pi*x/range[2]) + c[34]*sin(17*2*pi*x/range[2]) + c[35]*cos(17*2*pi*x/range[2]) +
    c[36]*sin(18*2*pi*x/range[2]) + c[37]*cos(18*2*pi*x/range[2]) + c[38]*sin(19*2*pi*x/range[2]) + c[39]*cos(19*2*pi*x/range[2]) +
    c[40]*sin(20*2*pi*x/range[2]) + c[41]*cos(20*2*pi*x/range[2]) + c[42]*sin(21*2*pi*x/range[2]) + c[43]*cos(21*2*pi*x/range[2]) +
    c[44]*sin(22*2*pi*x/range[2]) + c[45]*cos(22*2*pi*x/range[2]) + c[46]*sin(23*2*pi*x/range[2]) + c[47]*cos(23*2*pi*x/range[2]) +
    c[48]*sin(24*2*pi*x/range[2]) + c[49]*cos(24*2*pi*x/range[2]) + c[50]*sin(25*2*pi*x/range[2]) + c[51]*cos(25*2*pi*x/range[2])
  
  # Returning
  return(value)
  
}

# RMSE Difference
rmse_func <- function(x, c1, c2){
  result = ((final_beta_fourier(x, c1, c(0, 1)) - final_beta_fourier(x, c2, c(0, 1)))^2)
}


#############################

# Check 1