# GBMbaseline 0.2.0 (This package is not maintained anymore. Please use [RMV2.0](https://github.com/LBNL-ETA/RMV2.0) package instead)
*An R package for generating baseline models of electric load using Gradient Boosting Machine algorithm*

## Installation

To install GBMbaseline package you need first to install [R](https://cran.r-project.org/) and [RStudio](https://www.rstudio.com/).
Once you have R and RStudio installed, open Rstudio and install 'devtools' package:
```r
install.packages("devtools")
```
 Install the GBMbaseline package using the following command:

```r
devtools::install_github("samirtouzani/GBMbaseline")
```

## Package Description

The GBMbaseline package is aimed to provide an easy way to generate a baseline models using the Gradient Boosting Algorithm, and compute the statistical metrics that are relevant for comparing actual electric load with predicted one. For a more detailed description of the GBM baseline model the reader can refer to:
+ Touzani S. et al., [Gradient boosting machine for modeling the energy consumption of commercial buildings]()

## Input Data
The GBMbaseline package requires a specific format for the the csv file that corresponds to the input data. As an example of this format see bellow:

```r
"time","eload","Temp"
"1/1/14 1:00",24.17,52.6
"1/1/14 1:15",24.95,52.1
```
which correspond to the following table:

| time        | eload | Temp |
| ------------|:-----:| ----:|
| 1/1/14 23:00 | 24.17 | 52.6 |
| 1/1/14 23:15 | 24.95 | 52.1 |

Where "time" is the column of the timestamp, "eload" is the electric load and "Temp" is the outside air temperature. These three columns are required to produce a baseline model. If more variables are available, it's possible to add them to the model by adding them as new columns in the input data files and provide their names to the algorithm (this will be discussed later).

 Note that within the "GBMbaseline/data" folder two example files are provided.

#### Timestamps format

For convenience, an R function is included with the package to convert the actual timestamps format into "\%m/\%d/\%y \%H:\%M" format. To do so one should use the timestamp formatting used by the base R function *strptime* to define the format. For example converting a table that include a time column with the following timestamp format "2013-08-01 00:00:00" into a table with "1/8/13 00:00" format:

```r
data <- time_format(data,"%Y-%m-%d %H:%M:%S")
```

#### Holidays/Vacations Periods
The GBMbaseline package includes a mechanism for creating an additional variable that differentiate Holidays and vacation days from the rest of the data. To do so a csv file need to be provided. The format of this csv file is defined as follow:

```r
date
"2007/7/4"
"2007/9/3"
"2007/10/8"
```
An example corresponding to the US federal holidays is provided in “GBMbaseline/data” folder.


## Generarting the baseline models

The GBMbaseline package has the *gbm_baseline* function that reads the input data clean it from the missing values and outliers (for more details see the help of functions *clean_eload* and *clean_Temp*), build a GBM baseline model and return an gbm_baseline object.
```r
gbm_res <- gbm_baseline(train_path = "train_LBNL.csv",
                        pred_path = "pred_LBNL.csv",
                        days_off_path = "USA_Fed_Holidays.csv",
                        variables = c("Temp","tow")
                        )
```
*train_path* and *pred_path* are the paths of the input data files of respectively training period and prediction period.  *days_off_path* is the path of the file that contain the holidays and vacation days. The argument variables correspond to the variables that will be considered by the function as input variables. In the above example the variables are the temperature (*Temp*) and the Time Of the Week (*tow*). Note that since the *days_off_path* is provided the algorithm will add automatically a third variable, which is named *"days_off"* that correspond to the US federal holidays. If additional variables are considered the user needs to add the names of these variables to the R vector of the *variables* argument. For example if data of the solar radiation are available the user will need to add this data to the training and prediction data files as new column and name it, for example *solar_rad* then the *variables* argument needs to be modified into *c("Temp","tow","solar_rad")*.

#### Goodness of Fit Statistics

Once the GBM model is built, the user can access to the goodness of fit statistics using the following command:

```r
gbm_res$goodness_of_fit
```

#### Prediction Accuracy Statistics
To estimate the the prediction accuracy metrics the GBMbaseline package provide has a function:

```r
pred_accuracy(gbm_res)
```

#### Plots

The GBMbaseline package also provides three functions that allows the user to plot the time series of the actual and the predicted data. The first function *gbm_plot* will displays the time series for the training and the prediction periods. While *gbm_fit_plot* and *gbm_pred_plot* will separately display the data for the training period and the prediction period.

```r
gbm_plot(gbm_res)
gbm_fit_plot(gbm_res)
gbm_pred_plot(gbm_res)
```
