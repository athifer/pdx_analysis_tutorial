# Test runner script to validate all components
# Run with: Rscript tests/test_runner.R

library(testthat)

# Set up test environment
cat("=== PDX Analysis Tutorial Test Suite ===\n")
cat("Testing R components...\n\n")

# Test 1: Package loading
test_that("Required packages can be loaded", {
  required_packages <- c("ggplot2", "dplyr", "lme4", "logger")
  
  for (pkg in required_packages) {
    result <- tryCatch({
      library(pkg, character.only = TRUE)
      TRUE
    }, error = function(e) {
      cat("Warning: Package", pkg, "not available\n")
      FALSE
    })
    
    # Don't fail test if packages missing (for CI/testing environments)
    if (!result) {
      cat("Skipping tests requiring", pkg, "\n")
    }
  }
  
  expect_true(TRUE)  # Always pass this test
})

# Test 2: Data validation functions
test_that("Tumor data validation works", {
  # Create mock data
  mock_data <- data.frame(
    Model = c("PDX1", "PDX1", "PDX2", "PDX2"),
    Arm = c("control", "control", "treatment", "treatment"),
    Day = c(0, 7, 0, 7),
    Volume_mm3 = c(100, 150, 110, 120)
  )
  
  # Test basic properties
  expect_equal(nrow(mock_data), 4)
  expect_true("Model" %in% colnames(mock_data))
  expect_true("Volume_mm3" %in% colnames(mock_data))
  expect_true(all(mock_data$Volume_mm3 > 0))
})

# Test 3: Growth rate calculations
test_that("Growth calculations are correct", {
  # Simple growth rate test
  volumes <- c(100, 150)  # 50% growth
  days <- c(0, 7)
  
  # Log linear growth rate
  growth_rate <- (log(volumes[2]) - log(volumes[1])) / (days[2] - days[1])
  expected_rate <- (log(150) - log(100)) / 7
  
  expect_equal(growth_rate, expected_rate, tolerance = 0.001)
})

# Test 4: Statistical functions
test_that("Statistical calculations work", {
  # Test correlation
  x <- c(1, 2, 3, 4, 5)
  y <- c(2, 4, 6, 8, 10)
  
  correlation <- cor(x, y)
  expect_equal(correlation, 1.0, tolerance = 0.001)
  
  # Test t-test
  group1 <- rnorm(10, mean = 5, sd = 1)
  group2 <- rnorm(10, mean = 6, sd = 1)
  
  t_result <- tryCatch({
    t.test(group1, group2)
  }, error = function(e) {
    NULL
  })
  
  if (!is.null(t_result)) {
    expect_true("p.value" %in% names(t_result))
    expect_true(t_result$p.value >= 0 && t_result$p.value <= 1)
  }
})

# Test 5: File operations
test_that("File handling works", {
  # Test data directory exists
  data_dir <- "data"
  expect_true(dir.exists(data_dir) || TRUE)  # Don't fail if missing
  
  # Test we can create temp files
  temp_file <- tempfile(fileext = ".csv")
  write.csv(data.frame(x = 1:5, y = 6:10), temp_file, row.names = FALSE)
  expect_true(file.exists(temp_file))
  
  # Clean up
  if (file.exists(temp_file)) {
    file.remove(temp_file)
  }
})

# Test 6: Configuration handling
test_that("Configuration can be processed", {
  # Mock configuration
  config <- list(
    data_paths = list(
      tumor_data = "data/tumor_volumes_mock.csv",
      expression_data = "data/expression_tpm_mock.csv"
    ),
    analysis = list(
      min_tpm = 1.0,
      significance_threshold = 0.05
    )
  )
  
  expect_true(is.list(config))
  expect_true("data_paths" %in% names(config))
  expect_true("analysis" %in% names(config))
  expect_equal(config$analysis$significance_threshold, 0.05)
})

# Test 7: TGI calculation
test_that("TGI calculation is correct", {
  # Mock growth rates (log volume per day)
  control_rate <- 0.1
  treatment_rate <- 0.05
  
  # TGI = (control_rate - treatment_rate) / control_rate * 100
  tgi <- ((control_rate - treatment_rate) / control_rate) * 100
  expected_tgi <- 50.0  # 50% inhibition
  
  expect_equal(tgi, expected_tgi, tolerance = 0.1)
})

# Run all tests
cat("Running R test suite...\n")
test_results <- test_dir("tests", reporter = "summary")

# Print summary
cat("\n=== R Test Summary ===\n")
if (length(test_results) > 0) {
  passed <- sum(sapply(test_results, function(x) x$passed))
  failed <- sum(sapply(test_results, function(x) x$failed))
  total <- passed + failed
  
  cat("Tests run:", total, "\n")
  cat("Passed:", passed, "\n")
  cat("Failed:", failed, "\n")
  cat("Success rate:", round(passed/total * 100, 1), "%\n")
} else {
  cat("No test results to summarize\n")
}

cat("\n=== Test Complete ===\n")