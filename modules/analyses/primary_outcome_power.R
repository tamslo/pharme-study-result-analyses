print(R.version.string)

# install.packages('PowerTOST') # nolint: commented_code_linter.
library(PowerTOST)

# From ?sampleN.noninf:

# The estimated sample size gives always the total number of subjects (not
# subject/sequence in crossovers or subjects/group in parallel designs – like in
# some other software packages).

# theta0: ‘True’ or assumed T/R ratio or difference.
# In case of ‘logscale=TRUE’ it must be given as ratio T/R.
# If ‘logscale=FALSE’, the difference in means. In this case,
# the difference may be expressed in two ways: relative to the
# same (underlying) reference mean, i.e. as (T-R)/R = T/R - 1;
# or as difference in means T-R. Note that in the former case
# the units of ‘margin’ and ‘CV’ need also be given relative to
# the reference mean (specified as ratio).
# Defaults to 0.95 if ‘logscale=TRUE’ or to -0.05 if
# ‘logscale=FALSE’

# margin:

# If the supplied margin is < 0 (logscale=FALSE) or < 1 (logscale=TRUE), then
# it is assumed higher response values are better.

# CV (coefficient of variation):
# * logscale=TRUE the (geometric) coefficient of variation given as ratio
# * logscale=FALSE (residual) standard deviation of the responses
# * Cross-over studies within-subject CV,
# * Parallel-group design the CV of the total variability


print("Non-inferiority power analysis:")
print(paste("PowerTOST version", packageVersion("PowerTOST")))

results <- read.csv("data/comprehension_scores.csv")
redcap_data <- read.csv("data/external/redcap_data.csv")

test_group <- c()
reference_group <- c()
for (i in seq_len(nrow(results))) {
  row <- results[i, ]
  participant_id <- row[["participant_id"]]
  score <- row[["score"]]
  study_group <- redcap_data[
    which(redcap_data["participant_id"] == participant_id),
  ][["study_group"]]
  if (study_group == "PharMe") {
    test_group <- c(test_group, score)
  } else {
    reference_group <- c(reference_group, score)
  }
}

difference_in_means <- mean(test_group) - mean(reference_group)
standard_deviation <- sd(c(test_group, reference_group))
margin <- 0.1 * mean(reference_group)
print(paste("Actual theta0:", difference_in_means))
print(paste("Actual CV:", standard_deviation))
print(paste("10% margin:", margin))
power <- power.noninf(
  alpha = 0.025,
  n = c(length(test_group), length(reference_group)),
  design = "parallel",
  logscale = FALSE,
  theta0 = difference_in_means,
  margin = -margin,
  CV = standard_deviation,
)
print(paste("Power:", power))

# Do simple t-test power estimation for comparison (to check whether we are
# doing the power estimation correctly)

print("")

#install.packages("pwr") # nolint: commented_code_linter.
library(pwr)

print("Simple t-test power analysis:")
print(paste("pwr version", packageVersion("pwr")))

# Cohen's d: difference in means divided by the pooled standard deviation
reference_mean_with_margin <- mean(reference_group) - margin
difference_in_means_w_margin <- reference_mean_with_margin - mean(test_group)
n1 <- length(reference_group)
s1 <- sd(reference_group)
n2 <- length(test_group)
s2 <- sd(test_group)
# From https://www.geeksforgeeks.org/r-machine-learning/how-to-calculate-
# pooled-standard-deviation-in-r/
pooled_std <- sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n1 - 2))
effect_size <- difference_in_means_w_margin / pooled_std
print(paste("Effect size:", effect_size))

power_test <- pwr.t2n.test(
  n1 = n1,
  n2 = n2,
  d = effect_size,
  sig.level = 0.025,
  alternative = "less",
)

print(power_test)

print("Mann-Whitney-U power analysis:")

# Requires on Mac (brew): cmake, udunits, fftw
# install.packages('MKpower', dependencies = TRUE) # nolint: commented_code_linter, line_length_linter.

library(MKpower)

rerun_power_analysis <- FALSE

if (rerun_power_analysis) {
  print(paste("MKpower version", packageVersion("MKpower")))
  reference_group_minus_margin <- reference_group * 0.9

  wilcox_power <- sim.power.wilcox.test(
    nx = length(reference_group),
    ny = length(test_group),
    rx = function(n) {
      sample(reference_group_minus_margin, n, replace = TRUE)
    },
    ry = function(n) {
      sample(test_group, n, replace = TRUE)
    },
    rx.H0 = function(n) {
      sample(reference_group_minus_margin, n, replace = TRUE)
    },
    ry.H0 = function(n) {
      sample(reference_group_minus_margin, n, replace = TRUE)
    },
    alternative = "less",
    sig.level = 0.025,
    ties = TRUE,
    parallel = "multicore",
    ncpus = 8,
  )
  print(wilcox_power)
} else {
  # Output of code above, which takes about 15m 30s min to execute
  print("⚠️ Make sure to run analysis again if data changes!")
  print("Last run: January 13th, 2026")
  print("MKpower version 1.1")
  print("")
  print("Simulation Set-up")
  print("         nx = 86")
  print("         rx = function (n) , {,     return(sample(reference_group_minus_margin, n, replace = TRUE)), }") # nolint: line_length_linter.
  print("      rx.H0 = function (n) , {,     return(sample(reference_group_minus_margin, n, replace = TRUE)), }") # nolint: line_length_linter.
  print("         ny = 84")
  print("         ry = function (n) , {,     return(sample(test_group, n, replace = TRUE)), }") # nolint: line_length_linter.
  print("      ry.H0 = function (n) , {,     return(sample(reference_group_minus_margin, n, replace = TRUE)), }") # nolint: line_length_linter.
  print("  sig.level = 0.025")
  print("         mu = 0")
  print("alternative = less")
  print("       iter = 10000")
  print("   conf.int = FALSE")
  print("approximate = FALSE")
  print("       ties = TRUE")
  print("")
  print("Exact Wilcoxon-Mann-Whitney Test")
  print("       emp.power = 0.9704")
  print("emp.type.I.error = 0.0240")
  print("")
  print("Asymptotic Wilcoxon-Mann-Whitney Test")
  print("       emp.power = 0.9703")
  print("emp.type.I.error = 0.0240")
}