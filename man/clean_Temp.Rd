% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gbm_baseline.R
\name{clean_Temp}
\alias{clean_Temp}
\title{Clean the Temperature data}
\usage{
clean_Temp(Data, maxT = 130, minT = -80)
}
\arguments{
\item{maxT}{A numeric that correspond to the temperature above which the corresponding
observations will be excluded}

\item{minT}{A numeric that correspond to the temperature below which the corresponding
observations will be excluded}

\item{data}{A dataframe of training or prediction data.}
}
\value{
A dataframe that correspond to the cleaned training or prediction data
}
\description{
\code{clean_Temp} This function remove the observations which have Temperature values higher or lower
 than some predefined extreme values.
}
