#-------------------------------------------------------------------------------
#
#     Gradient Boosting machine baseline model function
#
#-------------------------------------------------------------------------------
#' Gradient boosting machine baseline model function.
#'
#' \code{gbm_baseline} This function build a baseline model using gradient boosting machine algorithm.
#'
#'
#' @param train_path The path of the file from which the training data are to be read.
#' @param pred_path The path of the file from which the prediction data are to be read.
#' @param days_off_path The path of the file from which the date data of days off (e.g., holidays) are to be read.
#' @param k_folds An integer that corresponds to the number of CV folds.
#' @param variables A vector that contains the names of the variables that will be considered by the function
#' as input variables.
#' @param ncores Number of threads used for the parallelization of the cross validation.
#' @param cv_blocks type of blocks for the cross validation; Default is "none", which correspond
#' to the standard k-fold cross validation technique.
#' @param iter The search grid combination of the number of iterations.
#' @param depth The search grid combination of the maximum depths.
#' @param lr The search grid combination of the learning rates.
#' @param subsample.
#'
#' @return a gbm_baseline object, which alist with the following components:
#' \describe{
#'   \item{gbm_model}{an object that has been created by the function xgboost,
#'    and which correspond to the optimal gbm model.}
#'   \item{train}{a dataframe that correspond to the training data after the
#'   cleaning and filtering function were applied.}
#'   \item{fitting}{the fitted values.}
#'   \item{goodness_of_fit}{a dataframe that contains the goodness of fitting metrics.}
#'   \item{gbm_cv_results}{a dataframe the training accuracy metrics (R2,
#'   RMSE and CVRMSE) and values of the tuning hype-parameters.}
#'   \item{tuned_parameters}{a list of the best hyper-parameters}
#'   \item{pred}{a dataframe that correspond to the prediction data after the
#'   cleaning and filtering function were applied.}
#'   \item{prediction}{the predicted values.}
#' }
#' @import dplyr
#' @import dygraphs
#' @export

gbm_baseline <- function(train_path,
                         pred_path = NULL,
                         days_off_path = NULL,
                         k_folds=5,
                         variables = c("Temp","tow"),
                         ncores = parallel::detectCores(logical =F),
                         cv_blocks = "days",
                         iter = (2:12)*25,
                         depth = c(3:7),
                         lr = c(0.05,0.1),
                         subsample=c(0.5)){
  train <- read.csv(file = train_path, header=T, row.names = NULL, stringsAsFactors = F)
  train <- time_var(train)
  if (!is.null(days_off_path)) {
   train <- days_off_var(days_off_path,train)
   variables <- c(variables,"days_off")
  }
  train <- clean_Temp(train)
  train <- clean_eload(train)

  cat('"================================="\n')
  cat('"Model Tuning"\n')
  cat('"================================="\n')

  tune_results <- gbm_tune(train,
                           k_folds = k_folds,
                           variables = variables,
                           ncores = ncores,
                           cv_blocks = cv_blocks,
                           iter = iter,
                           depth = depth,
                           lr = lr,
                           subsample = subsample)

  tuned_parameters <- tune_results$tuned_parameters
  gbm_cv_results <- tune_results$grid_results

  # Final gbm model
  train_output <- train$eload
  train_input <- train[,variables]
  cat('"================================="\n')
  cat('"Final Model Training"\n')
  cat('"================================="\n')
  gbm_model <- xgboost::xgboost(data = as.matrix(train_input),
                                label = train_output,
                                max_depth = tuned_parameters$best_depth,
                                eta = tuned_parameters$best_lr,
                                nrounds = tuned_parameters$best_iter,
                                subsample = tuned_parameters$best_subsample,
                                verbose = 0,
                                nthread = 1,
                                save_period = NULL)

  # Fitting:
  y_fit <- predict(gbm_model, as.matrix(train_input))
  fit_residuals <- train_output - y_fit

  goodness_of_fit <- as.data.frame(matrix(nr=1,nc=3))
  names(goodness_of_fit) <- c("fit_R2","fit_CVRMSE","fit_NMBE")
  goodness_of_fit$fit_R2 <- 100*(1-mean((fit_residuals)^2)/var(train_output))
  goodness_of_fit$fit_CVRMSE <- 100*sqrt(mean((fit_residuals)^2))/mean(train_output)
  goodness_of_fit$fit_NMBE <- 100*mean((fit_residuals))/mean(train_output)

  res <- NULL
  res$gbm_model <- gbm_model
  res$train <- train
  res$fitting <- y_fit
  res$goodness_of_fit <- goodness_of_fit
  res$gbm_cv_results <- gbm_cv_results
  res$tuned_parameters <- tuned_parameters

  # Prediction:
  if (!is.null(pred_path)) {
   cat('"================================="\n')
   cat('"Prediction"\n')
   cat('"================================="\n')
    pred <- read.csv(file = pred_path, header=T, row.names = NULL, stringsAsFactors = F)
    pred <- time_var(pred)
    if (!is.null(days_off_path)) {
     pred <- days_off_var(days_off_path,pred)
    }
    pred <- clean_Temp(pred)
    pred_input <- pred[,variables]
    y_pred <- predict(gbm_model, as.matrix(pred_input))
    res$pred <- pred
    res$prediction <- y_pred
  }
  return(res)
}



#-------------------------------------------------------------------------------
#
#     Gradient Boosting machine tuning function
#
#-------------------------------------------------------------------------------
#' Gradient boosting machine tuning function.
#'
#' \code{gbm_tune} splits the data into k folds by randomly selecting blocks of data,
#' where each block correspond to a calendar day.
#'
#'
#' @param Data A dataframe.
#' @param k_folds An integer that corresponds to the number of CV folds.
#' @param variables A vector that contains the names of the variables that will be considered by the function
#' as input variables.
#' @param ncores Number of threads used for the parallelization of the cross validation
#' @param cv_blocks type of blocks for the cross validation; Default is "none", which corresponds
#' to the standard cross validation technique
#' @param iter The search grid combination of the number of iterations
#' @param depth The search grid combination of the maximum depths
#' @param lr The search grid combination of the learning rates
#'
#' @return a list with the two following components:
#' \describe{
#'   \item{grid_results}{a dataframe the training accuracy metrics (R2,
#'   RMSE and CVRMSE) and values of the tuning hype-parameters }
#'   \item{tuned_parameters}{a list of the best hyper-parameters}
#' }
#'
#' @export

gbm_tune <- function(Data,k_folds,variables = c("Temp","tow"),
                     ncores, cv_blocks = "none",
                     iter, depth, lr,subsample){
  cl <- parallel::makeCluster(ncores)
  output <- Data$eload
  input <- Data[,variables]
  if (cv_blocks=="days"){
    list_train <- k_dblocks_cv(Data,k_folds)
  }
  if (cv_blocks=="weeks"){
    list_train <- k_wblocks_cv(Data,k_folds)
  }
  if (cv_blocks=="none"){
    list_train <- createFolds(y = output,k = k_folds,list = T)
  }
  gbm_grid <-  expand.grid(nrounds = iter,
                           max_depth = depth,
                           eta = lr,
                           subsample = subsample)

  n_grid <- dim(gbm_grid)[1]
  tab_grid_res <- data.frame(matrix(ncol = 10, nrow = n_grid))
  names(tab_grid_res) <- c("iter","depth","lr","subsample",
                           "R2","RMSE","CVRMSE",
                           "R2_sd","RMSE_sd","CVRMSE_sd")
  for (i in 1:n_grid){
    nrounds_i <- gbm_grid$nrounds[i]
    max_depth_i <- gbm_grid$max_depth[i]
    eta_i <- gbm_grid$eta[i]
    subsample_i <- gbm_grid$subsample[i]
    print(gbm_grid[i,])
    list_res <- parallel::parLapply(cl,
                                    list_train,
                                    gbm_cv_parallel,
                                    as.matrix(input),
                                    output,
                                    nrounds_i,
                                    max_depth_i,
                                    eta_i,
                                    subsample_i)
    tab_cv_res <- do.call("rbind", list_res)
    tab_grid_res$iter[i] <- nrounds_i
    tab_grid_res$depth[i] <- max_depth_i
    tab_grid_res$lr[i] <- eta_i
    tab_grid_res$subsample[i] <- subsample_i
    tab_grid_res$R2[i] <- mean(tab_cv_res$R2)
    tab_grid_res$RMSE[i] <- mean(tab_cv_res$RMSE)
    tab_grid_res$CVRMSE[i] <- mean(tab_cv_res$CVRMSE)
    tab_grid_res$R2_sd[i] <- sd(tab_cv_res$R2)
    tab_grid_res$RMSE_sd[i] <- sd(tab_cv_res$RMSE)
    tab_grid_res$CVRMSE_sd[i] <- sd(tab_cv_res$CVRMSE)
  }
  idx_best_param <- which(tab_grid_res$RMSE == min(tab_grid_res$RMSE))
  best_param <- list(best_iter = tab_grid_res$iter[idx_best_param],
                     best_depth = tab_grid_res$depth[idx_best_param],
                     best_lr = tab_grid_res$lr[idx_best_param],
                     best_subsample = tab_grid_res$subsample[idx_best_param])
  res <- NULL
  res$grid_results <- tab_grid_res
  res$tuned_parameters <- best_param
  return(res)
}


#' @export
gbm_cv_parallel <- function(idx_train,input,output,nrounds,max_depth,eta,subsample){
  tab_res <- as.data.frame(matrix(nr=1,nc=3))
  names(tab_res) <- c("R2","RMSE","CVRMSE")
  train <- input[-idx_train,]
  train_output <- output[-idx_train]
  test <- input[idx_train,]
  test_output <- output[idx_train]
  xgb_fit <- xgboost::xgboost(data = train,
                              label = train_output,
                              max_depth = max_depth,
                              eta = eta,
                              nrounds = nrounds,
                              objective = "reg:linear",
                              alpha=0,
                              colsample_bytree=1,
                              subsample=subsample,
                              verbose = 0,
                              nthread = 1,
                              save_period = NULL)
  yhat <- predict(xgb_fit, test)
  tab_res$R2[1] <- 1-mean((yhat - test_output)^2)/var(test_output)
  tab_res$RMSE[1] <- sqrt(mean((yhat - test_output)^2))
  tab_res$CVRMSE[1] <- 100*sqrt(mean((yhat - test_output)^2))/mean(test_output)
  return(tab_res)
}




#-------------------------------------------------------------------------------
#
#     K-dblocks-CV
#
#-------------------------------------------------------------------------------

#' K-fold-day cross validation function.
#'
#' \code{k_dblocks_cv} splits the data into k folds by randomly selecting blocks of data,
#' where each block correspond to a calendar day.
#'
#'
#' @param Data A dataframe.
#' @param k_folds An integer that corresponds to the number of CV folds.
#'
#' @return A list of row indexes corresponding to the training data.
#'
#' @export

k_dblocks_cv <- function(Data,k_folds){
  dates <- unique(Data$date)
  list_blocks_dates <- caret::createFolds(y = dates,k = k_folds,list = T)
  list_blocks_train <- list()
  for (i in 1:length(list_blocks_dates)){
    train_i <- NULL
    dates_i <- dates[list_blocks_dates[[i]]]
    for (j in 1:length(dates_i)){
      date_j <- dates_i[j]
      train_i <- c(train_i,which(Data$date == date_j))
    }
    list_blocks_train[[i]] <- train_i
  }
  return(list_blocks_train)
}

#-------------------------------------------------------------------------------
#
#     K-wblocks-CV
#
#-------------------------------------------------------------------------------
#' K-fold-week cross validation function.
#'
#' \code{k_wblocks_cv} splits the data into k folds by randomly selecting blocks of data,
#' where each block correspond to a calendar week.
#'
#'
#' @param Data A dataframe.
#' @param k_folds An integer that corresponds to the number of CV folds.
#'
#' @return A list of row indexes corresponding to the training data.
#'
#' @export

k_wblocks_cv <- function(Data,k_folds){
  nweeks <- unique(Data$week)
  list_blocks_weeks <- caret::createFolds(y = nweeks,k = k_folds,list = T)
  list_blocks_train <- list()
  for (i in 1:length(list_blocks_weeks)){
    train_i <- NULL
    weeks_i <- nweeks[list_blocks_weeks[[i]]]
    for (j in 1:length(weeks_i)){
      dweek_j <- weeks_i[j]
      train_i <- c(train_i,which(Data$week == dweek_j))
    }
    list_blocks_train[[i]] <- train_i
  }
  return(list_blocks_train)
}



#-------------------------------------------------------------------------------
#
#     Data preprocessing
#
#-------------------------------------------------------------------------------

#' Create new variables based on the timestamps.
#'
#' \code{time_variables} splits the data into k folds by randomly selecting blocks of data,
#' where each block correspond to a calendar day.
#'
#'
#' @param Data A dataframe of training or prediction data.
#' @param k_folds An integer that corresponds to the number of CV folds.
#'
#' @return A dataframe of training or prediction data including the new time variable.
#'
#' @export

time_var <- function(Data){
  dts <- as.POSIXct(strptime(Data$time, format = "%m/%d/%y %H:%M"))
  Data$dts <- dts
  Data$month <- lubridate::month(Data$dts)
  Data$wday <- as.POSIXlt(Data$dts)$wday
  Data$hour <- lubridate::hour(Data$dts) +1
  Data$minute <- lubridate::minute(Data$dts)
  Data$tod <- Data$hour + lubridate::minute(Data$dts)/60
  Data$tow <- Data$hour + lubridate::minute(Data$dts)/60 + Data$wday*24
  Data$date <- as.Date(dts)
  Data$week <- lubridate::week(dts)
  Data <- Data[complete.cases(Data),]
  return(Data)
}


#' Create a new binary variable based on the dates of days off (e.a., holidays).
#'
#' \code{days_off_var} This function create a new binary variable that correspond to the dates of days off,
#' which holidays or days when the building is not occupied.
#'
#'
#' @param days_off_path The path of the file from which the date data of days off (e.g., holidays) are to be read.
#' @param Data A dataframe of training or prediction data.
#'
#' @return A dataframe of training or prediction data including the new varable that correspond
#' to the days off.
#'

days_off_var <- function(days_off_path,Data){
  days_off<-read.csv(days_off_path,header =T)
  dts <- as.POSIXct(strptime(days_off$date, format = "%Y/%m/%d"))
  h_dts <- as.Date(dts)
  Data$days_off <- 0
  Data$days_off[Data$date %in% h_dts] <- 1
  return(Data)
}


#' Clean the elaod data
#'
#' \code{clean_eload} This function remove the observations that have negative eload values or
#' values higher or lower than some upper and lower thresholds. The upper threshold is defined as
#' \emph{n} percent higher than the quantile corresponding to the given upper probability \emph{U_ptresh} and the lower
#'  threshold is defined as \emph{n} percent lower than the quantile corresponding to the given lower probability \emph{L_ptresh}
#'  divided by \emph{n}.
#'
#'
#' @param Data A dataframe of training or prediction data.
#' @param n An integer that correspond to a multiplicative coefficient that is used
#' to define the thresholding value. The default value is 0.2 which correspond to \emph{20} percent.
#' @param L_ptresh A numeric that correspond to the probability of the lower quantile used for filtering
#' @param U_ptresh A numeric that correspond to the probability of the upper quantile used for filtering
#'

#' @return A dataframe that correspond to the cleaned data
#'
#' @export

clean_eload <- function(Data,n = .2, L_ptresh = 0.005, U_ptresh = 0.995){
  # exclude negative values
  Data <- filter(Data, eload >= 0)
  # exclude values higher than n*ptresh
  U_tresh <- as.numeric(quantile(Data$eload,probs = c(U_ptresh),na.rm = T))
  Data <- filter(Data, eload < (1+n)*U_tresh)
  # exclude the observations when the eload is n times lower than L_tresh
  L_tresh <- as.numeric(quantile(Data$eload,probs = c(L_ptresh),na.rm = T))
  Data <- filter(Data, eload > (L_tresh-(L_tresh*n)))
  return(Data)
}

#' Clean the Temperature data
#'
#' \code{clean_Temp} This function remove the observations which have Temperature values higher or lower
#'  than some predefined extreme values.
#'
#'
#' @param data A dataframe of training or prediction data.
#' @param maxT A numeric that correspond to the temperature above which the corresponding
#' observations will be excluded
#' @param minT A numeric that correspond to the temperature below which the corresponding
#' observations will be excluded

#' @return A dataframe that correspond to the cleaned training or prediction data
#'
#' @export

clean_Temp <- function(Data, maxT = 130, minT= -80){
  Data <- filter(Data, Temp <= maxT)
  Data <- filter(Data, Temp >= minT)
  return(Data)
}


#-------------------------------------------------------------------------------
#
#     Plot functions
#
#-------------------------------------------------------------------------------


#' Plot training period data
#'
#' \code{gbm_fit_plot} Read a gbm_baseline object and plots the actual and the fitted data
#'
#'
#' @param gbm_baseline_obj A gbm_baseline_obj

#' @return A dygraph time series plot
#'
#' @export

gbm_fit_plot <- function(gbm_baseline_obj){
  act <- select(gbm_baseline_obj$train,time,eload,Temp)
  dts <- as.POSIXct(strptime(act$time, format = "%m/%d/%y %H:%M"))
  act$dts <- dts
  if (length(which(duplicated(act$dts)==T)!=0)){
    act <- act[-which(duplicated(act$dts)==T),]
  }
  if (length(which(is.na(act$dts)))!=0){
    act <- act[-which(is.na(act$dts)),]
  }
  eload_xts<-xts::xts(act$eload, order.by = act$dts )
  names(eload_xts) <- c("eload")
  Temp_xts<-xts::xts(act$Temp, order.by = act$dts )
  names(Temp_xts) <- c("Temperature")
  data_xts <- cbind(eload_xts, Temp_xts)

  fit <- as.data.frame(gbm_baseline_obj$fitting)
  names(fit) <- c("eload")
  fit$dts <- dts
  if (length(which(duplicated(fit$dts)==T)!=0)){
    fit <- fit[-which(duplicated(fit$dts)==T),]
  }
  if (length(which(is.na(fit$dts)))){
    fit <- fit[-which(is.na(fit$dts)),]
  }
  eload_fit_xts<-xts::xts(fit$eload, order.by = fit$dts )
  names(eload_fit_xts) <- c("eload_fit")
  data_xts <- cbind(data_xts, eload_fit_xts)

  low_range <- min(abs(act$eload), na.rm = T)
  high_range <- max(act$eload, na.rm = T) * 1.6
  high_range_T <- max(act$Temp, na.rm = T) *1.1
  fit_metrics <- gbm_baseline_obj$goodness_of_fit
  title <- paste("goodness of fit:" , "(R2:",signif(fit_metrics$fit_R2, digits = 2),";",
                 "CVRMSE:",signif(fit_metrics$fit_CVRMSE, digits = 2),";",
                 "NMBE:",signif(fit_metrics$fit_NMBE, digits = 2),")",sep=" ")

  p<-dygraph(data_xts, main = title) %>% #
    dySeries("eload", label = "Actual", color = "#66C2A5") %>%
    dySeries("eload_fit", label = "Fitting", color = "#8DA0CB") %>%
    dyAxis("y", label = "eload", valueRange = c(low_range, high_range)) %>%
    dyAxis("y2", label = "Temperature", valueRange = c(-70, high_range_T))%>%
    dySeries("Temperature", axis = 'y2', label = "Temperature", color = "#EF3B2C")  %>%
    dyLegend(width = 350)
  return(p)
}



#' Plot prediction period data
#'
#' \code{gbm_pred_plot} Read a gbm_baseline object and plots the actual and the predicted data
#'
#'
#' @param gbm_baseline_obj A gbm_baseline object

#' @return A dygraph time series plot
#'
#' @export

gbm_pred_plot <- function(gbm_baseline_obj,title=NULL){
  act <- select(gbm_baseline_obj$pred,time,eload,Temp)
  dts <- as.POSIXct(strptime(act$time, format = "%m/%d/%y %H:%M"))
  act$dts <- dts
  if (length(which(duplicated(act$dts)==T)!=0)){
    act <- act[-which(duplicated(act$dts)==T),]
  }
  if (length(which(is.na(act$dts)))!=0){
    act <- act[-which(is.na(act$dts)),]
  }
  eload_xts<-xts::xts(act$eload, order.by = act$dts )
  names(eload_xts) <- c("eload")
  Temp_xts<-xts::xts(act$Temp, order.by = act$dts )
  names(Temp_xts) <- c("Temperature")
  data_xts <- cbind(eload_xts, Temp_xts)

  pred <- as.data.frame(gbm_baseline_obj$prediction)
  names(pred) <- c("eload")
  pred$dts <- dts
  if (length(which(duplicated(pred$dts)==T)!=0)){
    pred <- pred[-which(duplicated(pred$dts)==T),]
  }
  if (length(which(is.na(pred$dts)))){
    pred <- pred[-which(is.na(pred$dts)),]
  }
  eload_pred_xts<-xts::xts(pred$eload, order.by = pred$dts )
  names(eload_pred_xts) <- c("eload_pred")
  data_xts <- cbind(data_xts, eload_pred_xts)

  low_range <- min(abs(act$eload), na.rm = T)
  high_range <- max(act$eload, na.rm = T) * 1.6
  high_range_T <- max(act$Temp, na.rm = T) *1.1


  p<-dygraph(data_xts, main = title) %>%
    dySeries("eload", label = "Actual", color = "#66C2A5") %>%
    dySeries("eload_pred", label = "Prediction", color = "#FC8D62") %>%
    dyAxis("y", label = "eload", valueRange = c(low_range, high_range)) %>%
    dyAxis("y2", label = "Temperature", valueRange = c(-70, high_range_T))%>%
    dySeries("Temperature", axis = 'y2', label = "Temperature", color = "#EF3B2C")  %>%
    dyLegend(width = 350)
  return(p)
}



#' Plot training and prediction periods data
#'
#' \code{gbm_plot} Read a gbm_baseline object and plots the actual, the fitted and the predicted data
#'
#'
#' @param gbm_baseline_obj A gbm_baseline object

#' @return A dygraph time series plot
#'
#' @export


gbm_plot <- function(gbm_baseline_obj,title=NULL){
  act_train <- select(gbm_baseline_obj$train,time,eload,Temp)
  act_pred <- select(gbm_baseline_obj$pred,time,eload,Temp)
  act <- rbind(act_train,act_pred)
  dts <- as.POSIXct(strptime(act$time, format = "%m/%d/%y %H:%M"))
  act$dts <- dts
  act$dts <- as.POSIXct(strptime(act$time, format = "%m/%d/%y %H:%M"))
  if (length(which(duplicated(act$dts)==T)!=0)){
    act <- act[-which(duplicated(act$dts)==T),]
  }
  if (length(which(is.na(act$dts)))){
    act <- act[-which(is.na(act$dts)),]
  }
  eload_xts<-xts::xts(act$eload, order.by = act$dts )
  names(eload_xts) <- c("eload")
  Temp_xts<-xts::xts(act$Temp, order.by = act$dts )
  names(Temp_xts) <- c("Temperature")
  data_xts <- cbind(eload_xts, Temp_xts)

  fit <- as.data.frame(gbm_baseline_obj$fitting)
  names(fit) <- c("eload")
  fit$dts <- as.POSIXct(strptime(act_train$time, format = "%m/%d/%y %H:%M"))
  if (length(which(duplicated(fit$dts)==T)!=0)){
    fit <- fit[-which(duplicated(fit$dts)==T),]
  }
  if (length(which(is.na(fit$dts)))){
    fit <- fit[-which(is.na(fit$dts)),]
  }
  eload_fit_xts<-xts::xts(fit$eload, order.by = fit$dts )
  names(eload_fit_xts) <- c("eload_fit")
  data_xts <- cbind(data_xts, eload_fit_xts)

  pred <- as.data.frame(gbm_baseline_obj$prediction)
  names(pred) <- c("eload")
  pred$dts <- as.POSIXct(strptime(act_pred$time, format = "%m/%d/%y %H:%M"))
  if (length(which(duplicated(pred$dts)==T)!=0)){
    pred <- pred[-which(duplicated(pred$dts)==T),]
  }
  if (length(which(is.na(pred$dts)))){
    pred <- pred[-which(is.na(pred$dts)),]
  }
  eload_pred_xts<-xts::xts(pred$eload, order.by = pred$dts )
  names(eload_pred_xts) <- c("eload_pred")
  data_xts <- cbind(data_xts, eload_pred_xts)

  low_range <- min(abs(act$eload), na.rm = T)
  high_range <- max(act$eload, na.rm = T) * 1.6
  high_range_T <- max(act$Temp, na.rm = T) *1.1

  p<-dygraph(data_xts, main = title) %>%
    dySeries("eload", label = "Actual", color = "#66C2A5") %>%
    dySeries("eload_pred", label = "Prediction", color = "#FC8D62") %>%
    dySeries("eload_fit", label = "Fitting", color = "#8DA0CB") %>%
    dyAxis("y", label = "eload", valueRange = c(low_range, high_range)) %>%
    dyAxis("y2", label = "Temperature", valueRange = c(-70, high_range_T))%>%
    dySeries("Temperature", axis = 'y2', label = "Temperature", color = "#EF3B2C")  %>%
    dyLegend(width = 350)

  p <- plot_shading(gbm_baseline_obj,p)
  return(p)

}


plot_shading <- function(gbm_baseline_obj,gbm_plot_obj){
  pred <- select(gbm_baseline_obj$pred,time)
  pred$dts <- as.POSIXct(strptime(pred$time, format = "%m/%d/%y %H:%M"))
  start_pred <- min(pred$dts)
  end_pred <- max(pred$dts)
  gbm_plot_obj<-dyShading(gbm_plot_obj,from = start_pred, to = end_pred, color = "#FFE6E6")

  train <- select(gbm_baseline_obj$train,time)
  train$dts <- as.POSIXct(strptime(train$time, format = "%m/%d/%y %H:%M"))
  start_train <- min(train$dts)
  end_train <- max(train$dts)
  gbm_plot_obj<-dyShading(gbm_plot_obj,from = start_train, to = end_train, color = "#CCEBD6")#"#CCEBD6")
  return(gbm_plot_obj)
}


#' Convert the timestapms into the default format
#'
#' \code{time_format} This function convert the actual timestamps format into "\%m/\%d/\%y \%H:\%M" format
#'
#'
#' @param data A dataframe that contains time column
#' @param format A character string that define the actual format of the timestamps.
#' Use the description of the base R function \emph{strptime} to define the format.
#' @return A dataframe with timestamps converted into "\%m/\%d/\%y \%H:\%M" format
#'
#' @export


time_format <- function(data,format){
  if (!is.null(format)){
    stop("The original time format is not indicated")
  }
  dts <- as.POSIXct(strptime(data$time, format = format))
  data$time <- format(dts,"%m/%d/%y %H:%M")
}

#' Prediction accuracy metrics computation
#'
#' \code{pred_accuracy} This function compute the following prediction accuracy metrics:  R2, CV(RMSE) and NBME
#'
#'
#' @param gbm_baseline_obj A gbm_baseline object
#' @return A dataframe with the computer R2, CV(RMSE) and NMBE
#'
#' @export

pred_accuracy <- function(gbm_baseline_obj){
 y_pred <- gbm_baseline_obj$prediction
 pred_output <- select(gbm_baseline_obj$pred,eload)
 pred_residuals <- pred_output$eload - y_pred
 pred_metrics <- as.data.frame(matrix(nr=1,nc=3))
 names(pred_metrics) <- c("pred_R2","pred_CVRMSE","pred_NMBE")
 pred_metrics$pred_R2 <- 100*(1-mean((pred_residuals)^2)/var(pred_output$eload))
 pred_metrics$pred_CVRMSE <- 100*sqrt(mean((pred_residuals)^2))/mean(pred_output$eload)
 pred_metrics$pred_NMBE <- 100*mean((pred_residuals))/mean(pred_output$eload)
 return(pred_metrics)
}
