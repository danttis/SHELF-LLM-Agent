#!/usr/bin/env Rscript
# shelf_bridge.R
#
# Thin bridge script that reads JSON from stdin, calls SHELF
# functions, and writes JSON to stdout.
#
# This script does NOT modify any SHELF source code.
# It is called by the Python agent via subprocess.
#
# Supported actions:
#   - fitdist
#   - feedback
#   - sampleFit
#   - fitprecision

suppressPackageStartupMessages({
    library(SHELF)
    library(jsonlite)
})

# --- Helper: convert Inf strings back to R Inf ---
safe_number <- function(x) {
    if (is.character(x)) {
        if (x == "Inf") return(Inf)
        if (x == "-Inf") return(-Inf)
        return(as.numeric(x))
    }
    return(x)
}

# --- Read input ---
input <- fromJSON(file("stdin"), simplifyVector = TRUE)
action <- input$action
params <- input$params

# Set seed if provided
if (!is.null(input$seed)) {
    set.seed(input$seed)
}

# --- Dispatch actions ---

if (action == "fitdist") {

    vals  <- as.numeric(params$vals)
    probs <- as.numeric(params$probs)
    lower <- safe_number(params$lower)
    upper <- safe_number(params$upper)
    tdf   <- as.integer(params$tdf)

    fit <- fitdist(
        vals  = vals,
        probs = probs,
        lower = lower,
        upper = upper,
        tdf   = tdf
    )

    result <- list(
        Normal = list(
            mean = fit$Normal[1, "mean"],
            sd   = fit$Normal[1, "sd"]
        ),
        Student.t = list(
            location = fit$Student.t[1, "location"],
            scale    = fit$Student.t[1, "scale"],
            df       = fit$Student.t[1, "df"]
        ),
        Gamma = list(
            shape = fit$Gamma[1, "shape"],
            rate  = fit$Gamma[1, "rate"]
        ),
        Log.normal = list(
            mean_log_X = fit$Log.normal[1, "mean.log.X"],
            sd_log_X   = fit$Log.normal[1, "sd.log.X"]
        ),
        Beta = list(
            shape1 = fit$Beta[1, "shape1"],
            shape2 = fit$Beta[1, "shape2"]
        ),
        mirrorgamma = list(
            shape = fit$mirrorgamma[1, "shape"],
            rate  = fit$mirrorgamma[1, "rate"]
        ),
        mirrorlognormal = list(
            mean_log_X = fit$mirrorlognormal[1, "mean.log.X"],
            sd_log_X   = fit$mirrorlognormal[1, "sd.log.X"]
        ),
        ssq = as.list(fit$ssq[1, ]),
        best_fitting = as.character(fit$best.fitting[1, 1]),
        vals   = as.numeric(fit$vals),
        probs  = as.numeric(fit$probs),
        limits = list(
            lower = fit$limits[1, "lower"],
            upper = fit$limits[1, "upper"]
        )
    )

    cat(toJSON(result, auto_unbox = TRUE, na = "null"))

} else if (action == "feedback") {

    # Reconstruct the fit object from JSON
    fit_data <- params$fit
    vals  <- as.numeric(fit_data$vals)
    probs <- as.numeric(fit_data$probs)
    lower <- safe_number(fit_data$limits$lower)
    upper <- safe_number(fit_data$limits$upper)

    fit <- fitdist(
        vals  = vals,
        probs = probs,
        lower = lower,
        upper = upper
    )

    fb_quantiles <- as.numeric(params$quantiles)
    fb <- feedback(fit, quantiles = fb_quantiles, ex = 1)

    result <- list(
        fitted_quantiles = as.list(
            as.data.frame(fb$fitted.quantiles)
        ),
        fitted_probabilities = if (!is.null(fb$fitted.probabilities)) {
            as.list(as.data.frame(fb$fitted.probabilities))
        } else {
            NULL
        }
    )

    cat(toJSON(result, auto_unbox = TRUE, na = "null"))

} else if (action == "sampleFit") {

    fit_data <- params$fit
    vals  <- as.numeric(fit_data$vals)
    probs <- as.numeric(fit_data$probs)
    lower <- safe_number(fit_data$limits$lower)
    upper <- safe_number(fit_data$limits$upper)

    fit <- fitdist(
        vals  = vals,
        probs = probs,
        lower = lower,
        upper = upper
    )

    n <- as.integer(params$n)
    expert <- as.integer(params$expert)

    samples <- sampleFit(fit, n = n, expert = expert)
    result <- as.list(as.data.frame(samples))

    cat(toJSON(result, auto_unbox = TRUE, na = "null"))

} else if (action == "fitprecision") {

    interval <- as.numeric(params$interval)
    propvals <- as.numeric(params$propvals)
    propprobs <- as.numeric(params$propprobs)
    trans <- as.character(params$trans)
    tdf <- as.integer(params$tdf)

    extra_args <- list(
        interval  = interval,
        propvals  = propvals,
        propprobs = propprobs,
        trans     = trans,
        tdf       = tdf,
        pplot     = FALSE
    )

    if (!is.null(params$med)) {
        extra_args$med <- as.numeric(params$med)
    }

    fit <- do.call(fitprecision, extra_args)

    result <- list(
        Gamma = list(
            shape = fit$Gamma[1, "shape"],
            rate  = fit$Gamma[1, "rate"]
        ),
        Log.normal = list(
            mean_log_X = fit$Log.normal[1, "mean.log.X"],
            sd_log_X   = fit$Log.normal[1, "sd.log.X"]
        ),
        best_fitting = as.character(fit$best.fitting[1, 1]),
        ssq = as.list(fit$ssq[1, ]),
        interval  = as.numeric(fit$interval),
        transform = fit$transform
    )

    cat(toJSON(result, auto_unbox = TRUE, na = "null"))

} else {
    cat(toJSON(
        list(error = paste("Unknown action:", action)),
        auto_unbox = TRUE
    ))
    quit(status = 1)
}
