# Robust tumor volume analysis with error handling and logging
# Enhanced version of analyze_volume.R with better practices

# Load required libraries with error handling
required_packages <- c("DRAP", "lme4", "ggplot2", "configr", "logger")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    stop(paste("Required package", pkg, "is not installed. Please install it first."))
  }
}

# Setup logging
log_threshold(INFO)
log_appender(appender_file("logs/analysis.log"))

#' Load and validate tumor volume data
#' @param file_path Path to the tumor volume CSV file
#' @return Data frame with validated tumor volume data
load_tumor_data <- function(file_path) {
  log_info("Loading tumor volume data from: {file_path}")
  
  if (!file.exists(file_path)) {
    log_error("Data file not found: {file_path}")
    stop("Data file not found: ", file_path)
  }
  
  tryCatch({
    df <- read.csv(file_path, stringsAsFactors = FALSE)
    log_info("Successfully loaded {nrow(df)} rows of data")
    
    # Validate required columns
    required_cols <- c("Model", "Arm", "Day", "Volume_mm3")
    missing_cols <- setdiff(required_cols, names(df))
    
    if (length(missing_cols) > 0) {
      log_error("Missing required columns: {paste(missing_cols, collapse=', ')}")
      stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
    }
    
    # Data type validation
    df$Day <- as.integer(df$Day)
    df$Volume_mm3 <- as.numeric(df$Volume_mm3)
    df$Model <- as.factor(df$Model)
    df$Arm <- as.factor(df$Arm)
    
    # Remove rows with missing volume data
    initial_rows <- nrow(df)
    df <- df[!is.na(df$Volume_mm3) & df$Volume_mm3 > 0, ]
    removed_rows <- initial_rows - nrow(df)
    
    if (removed_rows > 0) {
      log_warn("Removed {removed_rows} rows with missing or invalid volume data")
    }
    
    log_info("Data validation completed successfully")
    return(df)
    
  }, error = function(e) {
    log_error("Error loading data: {e$message}")
    stop("Error loading data: ", e$message)
  })
}

#' Detect and handle outliers in tumor volume data
#' @param df Data frame with tumor volume data
#' @param threshold Z-score threshold for outlier detection (default: 3)
#' @return Data frame with outliers flagged
detect_outliers <- function(df, threshold = 3) {
  log_info("Detecting outliers with threshold: {threshold}")
  
  df$outlier <- FALSE
  
  for (model in levels(df$Model)) {
    model_data <- df[df$Model == model, ]
    
    if (nrow(model_data) > 3) {  # Need sufficient data points
      z_scores <- abs(scale(log(model_data$Volume_mm3 + 1)))
      outlier_indices <- which(z_scores > threshold)
      
      if (length(outlier_indices) > 0) {
        df[df$Model == model, "outlier"][outlier_indices] <- TRUE
        log_warn("Found {length(outlier_indices)} outliers in model {model}")
      }
    }
  }
  
  n_outliers <- sum(df$outlier)
  log_info("Total outliers detected: {n_outliers}")
  
  return(df)
}

#' Fit linear mixed-effects model for tumor growth
#' @param df Data frame with tumor volume data
#' @param exclude_outliers Logical, whether to exclude outliers from analysis
#' @return fitted lme4 model object
fit_growth_model <- function(df, exclude_outliers = TRUE) {
  log_info("Fitting linear mixed-effects model")
  
  if (exclude_outliers) {
    analysis_data <- df[!df$outlier, ]
    excluded_rows <- sum(df$outlier)
    log_info("Excluding {excluded_rows} outlier observations from analysis")
  } else {
    analysis_data <- df
  }
  
  # Log transformation with small constant to handle zeros
  analysis_data$logVol <- log(analysis_data$Volume_mm3 + 1)
  
  tryCatch({
    # Fit linear mixed model
    model_formula <- logVol ~ Arm * Day + (1 | Model)
    mod <- lmer(model_formula, data = analysis_data)
    
    log_info("Model fitted successfully")
    log_info("Number of observations: {nobs(mod)}")
    log_info("Number of groups (Models): {length(unique(analysis_data$Model))}")
    
    return(mod)
    
  }, error = function(e) {
    log_error("Error fitting model: {e$message}")
    stop("Error fitting model: ", e$message)
  })
}

#' Calculate Tumor Growth Inhibition (TGI)
#' @param df Data frame with tumor volume data
#' @param endpoint_day Day to calculate TGI (default: max day)
#' @return TGI value as percentage
calculate_tgi <- function(df, endpoint_day = NULL) {
  log_info("Calculating Tumor Growth Inhibition (TGI)")
  
  if (is.null(endpoint_day)) {
    endpoint_day <- max(df$Day)
  }
  
  # Get baseline (day 0) and endpoint data
  baseline <- df[df$Day == 0, ]
  endpoint <- df[df$Day == endpoint_day, ]
  
  if (nrow(baseline) == 0 || nrow(endpoint) == 0) {
    log_error("Missing baseline or endpoint data for TGI calculation")
    stop("Missing baseline or endpoint data for TGI calculation")
  }
  
  # Calculate mean volumes by treatment arm
  control_start <- mean(baseline[baseline$Arm == "control", "Volume_mm3"], na.rm = TRUE)
  control_end <- mean(endpoint[endpoint$Arm == "control", "Volume_mm3"], na.rm = TRUE)
  
  treatment_start <- mean(baseline[baseline$Arm == "treatment", "Volume_mm3"], na.rm = TRUE)
  treatment_end <- mean(endpoint[endpoint$Arm == "treatment", "Volume_mm3"], na.rm = TRUE)
  
  # Calculate TGI
  control_growth <- control_end - control_start
  treatment_growth <- treatment_end - treatment_start
  
  tgi <- 100 * (1 - treatment_growth / control_growth)
  
  log_info("TGI calculated: {round(tgi, 2)}% at day {endpoint_day}")
  
  return(tgi)
}

#' Main analysis function
main_analysis <- function(config_file = "config/config.ini") {
  log_info("Starting PDX tumor volume analysis")
  
  # Load configuration
  if (file.exists(config_file)) {
    config <- read.config(config_file)
    log_info("Configuration loaded from: {config_file}")
  } else {
    log_warn("Configuration file not found, using defaults")
    config <- list(
      data_paths = list(tumor_volumes = "data/tumor_volumes_mock.csv"),
      analysis_parameters = list(outlier_threshold = 3.0, tgi_endpoint_day = 28),
      statistical = list(random_seed = 42)
    )
  }
  
  # Set random seed for reproducibility
  set.seed(config$statistical$random_seed)
  
  # Load and process data
  df <- load_tumor_data(config$data_paths$tumor_volumes)
  df <- detect_outliers(df, config$analysis_parameters$outlier_threshold)
  
  # Fit statistical model
  model <- fit_growth_model(df)
  
  # Calculate TGI
  tgi <- calculate_tgi(df, config$analysis_parameters$tgi_endpoint_day)
  
  # Print results
  cat("\n=== TUMOR VOLUME ANALYSIS RESULTS ===\n")
  cat("Linear Mixed-Effects Model Summary:\n")
  print(summary(model))
  
  cat(sprintf("\nTumor Growth Inhibition (TGI): %.2f%%\n", tgi))
  
  log_info("Analysis completed successfully")
  
  return(list(model = model, tgi = tgi, data = df))
}

# Create logs directory if it doesn't exist
if (!dir.exists("logs")) {
  dir.create("logs", recursive = TRUE)
}

# Run analysis if script is executed directly
if (!interactive()) {
  results <- main_analysis()
}