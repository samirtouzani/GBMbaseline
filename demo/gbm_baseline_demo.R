rm(list = ls())
library(GBMbaseline)
set.seed(2014)


setwd("/Users/stouzani/Google\ Drive/BTUS_Projects/MV_Projects/My_load_model/GBM_R_Package/GBMbaseline/data/")

# Run the gbm baseline model

gbm_res <- gbm_baseline(train_path = "train_LBNL.csv",
                        pred_path = "pred_LBNL.csv",
                        days_off_path = "USA_Fed_Holidays.csv",
                        k_folds=5,
                        ncores=5,
                        variables = c("Temp","tow"),
                        cv_blocks = "days")

# Display the goodness of fit metrics

gbm_res$goodness_of_fit

# Display the prediction accuracy metrics

pred_accuracy(gbm_res)

# Plots
gbm_plot(gbm_res)
gbm_fit_plot(gbm_res)
gbm_pred_plot(gbm_res)

