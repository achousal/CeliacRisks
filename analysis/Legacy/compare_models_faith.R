#!/usr/bin/env Rscript
#============================================================
# compare_models_faith.R
#
# Key features:
#   - OOF-based model selection (NO test leakage)
#   - High-specificity comparison aligned to deployment:
#       * threshold set on TRAIN controls (controls_oof_mean.csv)
#       * performance evaluated on TEST (test_preds.csv)
#   - Paired-delta summaries vs winner with CIs (OOF per repeat)
#   - Test calibration summary including ECE + calibration RMSE
#   - High-specificity workload + TP/10k people
#   - Clinical summary table per scenario
#============================================================

suppressPackageStartupMessages({
  library(optparse)
  if (requireNamespace("readr", quietly = TRUE)) {
    library(readr)
  }
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(ggplot2)
  library(scales)
})

if (!exists("read_csv")) {
  message("[WARN] readr not installed; falling back to utils::read.csv")
  read_csv <- function(file, ...) {
    dots <- list(...)
    dots$show_col_types <- NULL
    dots$progress <- NULL
    if (is.null(dots$stringsAsFactors)) {
      dots$stringsAsFactors <- FALSE
    }
    do.call(utils::read.csv, c(list(file = file), dots))
  }
}

#-----------------------------#
# Logging / helpers
#-----------------------------#
msg  <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%F %T")), sprintf(...), "\n", sep = "")
wmsg <- function(...) warning(sprintf("[%s] ", format(Sys.time(), "%F %T")), sprintf(...), call. = FALSE)

dir_ok <- function(p) { dir.create(p, recursive = TRUE, showWarnings = FALSE); p }
exists_file <- function(path) !is.null(path) && nzchar(path) && file.exists(path)
exists_dir  <- function(path) !is.null(path) && nzchar(path) && dir.exists(path)

read_csv_or_warn <- function(path, what = "file", ...) {
  if (!exists_file(path)) { wmsg("Missing %s: %s", what, path); return(NULL) }
  tryCatch(read_csv(path, show_col_types = FALSE, ...),
           error = function(e) { wmsg("Failed reading %s: %s (%s)", what, path, e$message); NULL })
}

save_plot <- function(p, file, w = 8, h = 5, dpi = 300) {
  ggsave(file, p, width = w, height = h, dpi = dpi,
         bg = "white", scale = 1.0, limitsize = FALSE)
  msg("wrote: %s", file)
}

as_num <- function(x) suppressWarnings(as.numeric(x))

parse_ci_string <- function(ci) {
  s <- as.character(ci)
  m <- str_match(s, "\\[\\s*([0-9eE\\.+-]+)\\s*,\\s*([0-9eE\\.+-]+)\\s*\\]")
  tibble(lo = as_num(m[,2]), hi = as_num(m[,3]))
}

parse_panel_n <- function(model) {
  m <- stringr::str_match(model, "panelN(\\d+)")
  as.integer(m[,2])
}

#-----------------------------#
# Curve helpers
#-----------------------------#
pr_curve <- function(y, p) {
  y <- as.integer(y); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p)
  y <- y[ok]; p <- p[ok]
  o <- order(p, decreasing = TRUE)
  y <- y[o]; p <- p[o]
  P <- sum(y == 1)
  if (P == 0) stop("PR curve: no positives", call. = FALSE)
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  tibble(recall = tp / P,
         precision = tp / pmax(1, tp + fp),
         threshold = p) %>%
    group_by(threshold) %>% slice_tail(n = 1) %>% ungroup()
}

roc_curve <- function(y, p) {
  y <- as.integer(y); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p)
  y <- y[ok]; p <- p[ok]
  o <- order(p, decreasing = TRUE)
  y <- y[o]; p <- p[o]
  P <- sum(y == 1); N <- sum(y == 0)
  if (P == 0 || N == 0) stop("ROC curve: need both classes", call. = FALSE)
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  tibble(tpr = tp / P, fpr = fp / N, threshold = p) %>%
    group_by(threshold) %>% slice_tail(n = 1) %>% ungroup()
}

calibration_bins <- function(y, p, n_bins = 10) {
  y <- as.integer(y); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p)
  y <- y[ok]; p <- p[ok]
  if (length(y) == 0) return(tibble())
  b <- cut(p, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  tibble(bin = b, p = p, y = y) %>%
    group_by(bin) %>%
    summarize(prob_pred = mean(p), prob_true = mean(y), n = dplyr::n(), .groups = "drop") %>%
    arrange(prob_pred)
}

ece_from_bins <- function(cal_tbl) {
  if (is.null(cal_tbl) || nrow(cal_tbl) == 0) return(NA_real_)
  n_tot <- sum(cal_tbl$n)
  if (!is.finite(n_tot) || n_tot <= 0) return(NA_real_)
  sum((cal_tbl$n / n_tot) * abs(cal_tbl$prob_true - cal_tbl$prob_pred))
}

cal_rmse_from_bins <- function(cal_tbl) {
  if (is.null(cal_tbl) || nrow(cal_tbl) == 0) return(NA_real_)
  n_tot <- sum(cal_tbl$n)
  if (!is.finite(n_tot) || n_tot <= 0) return(NA_real_)
  sqrt(sum((cal_tbl$n / n_tot) * (cal_tbl$prob_true - cal_tbl$prob_pred)^2))
}

assess_calibration <- function(intercept, slope) {
  flags <- c()
  if (is.finite(intercept)) {
    if (abs(intercept) > 1.0) flags <- c(flags, "SEVERE_INTERCEPT")
    else if (abs(intercept) > 0.5) flags <- c(flags, "MODERATE_INTERCEPT")
  }
  if (is.finite(slope)) {
    if (slope < 0.7 || slope > 1.5) flags <- c(flags, "SEVERE_SLOPE")
    else if (slope < 0.85 || slope > 1.2) flags <- c(flags, "MODERATE_SLOPE")
  }
  if (length(flags) == 0) return("GOOD")
  paste(flags, collapse = "; ")
}

net_benefit <- function(y, p, pt) {
  y <- as.integer(y); p <- as.numeric(p)
  n <- length(y)
  if (pt <= 0 || pt >= 1 || n == 0) return(NA_real_)
  TP <- sum((p >= pt) & (y == 1))
  FP <- sum((p >= pt) & (y == 0))
  w <- pt / (1 - pt)
  (TP/n) - (FP/n) * w
}

#-----------------------------#
# High-specificity helpers
#-----------------------------#
threshold_for_spec <- function(p_ctrl, target_spec) {
  p_ctrl <- as.numeric(p_ctrl)
  p_ctrl <- p_ctrl[is.finite(p_ctrl)]
  if (length(p_ctrl) == 0) return(NA_real_)
  as.numeric(quantile(p_ctrl, probs = target_spec, type = 7))
}

metrics_at_threshold <- function(y, p, thr) {
  y <- as.integer(y)
  p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p)
  y <- y[ok]; p <- p[ok]

  if (length(y) == 0) {
    return(list(sens = NA_real_, spec = NA_real_, ppv = NA_real_,
                npv = NA_real_, fp = NA_integer_, tp = NA_integer_,
                fn = NA_integer_, tn = NA_integer_, n = 0L))
  }

  pred_pos <- p >= thr
  tp <- sum(pred_pos & (y == 1))
  fp <- sum(pred_pos & (y == 0))
  tn <- sum(!pred_pos & (y == 0))
  fn <- sum(!pred_pos & (y == 1))

  sens <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  ppv  <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  npv  <- if ((tn + fn) > 0) tn / (tn + fn) else NA_real_

  list(sens = sens, spec = spec, ppv = ppv, npv = npv,
       tp = tp, fp = fp, tn = tn, fn = fn, n = length(y))
}

#-----------------------------#
# Discover run dirs + find preds
#-----------------------------#
discover_run_dirs <- function(results_root) {
  d <- list.dirs(results_root, full.names = TRUE, recursive = FALSE)
  d <- d[basename(d) != "COMBINED"]
  keep <- d[file.exists(file.path(d, "core", "test_metrics.csv"))]
  sort(keep)
}

find_test_pred_file <- function(run_dirs, scen, model) {
  target <- sprintf("%s__test_preds__%s.csv", scen, model)
  for (rd in run_dirs) {
    f <- file.path(rd, "preds", "test_preds", target)
    if (file.exists(f)) return(f)
  }
  return(NA_character_)
}

find_controls_oof_file <- function(run_dirs, scen, model) {
  target <- sprintf("%s__controls_risk__%s__oof_mean.csv", scen, model)
  for (rd in run_dirs) {
    f <- file.path(rd, "preds", "controls_oof", target)
    if (file.exists(f)) return(f)
  }
  return(NA_character_)
}

collect_test_preds_all_runs <- function(run_dirs, scen, model) {
  target <- sprintf("%s__test_preds__%s.csv", scen, model)
  preds_list <- map(run_dirs, function(rd) {
    f <- file.path(rd, "preds", "test_preds", target)
    if (!file.exists(f)) return(NULL)
    d <- read_csv_or_warn(f, sprintf("test preds (%s/%s)", scen, model))
    if (is.null(d) || !all(c("y_true", "risk_test") %in% names(d))) return(NULL)
    adj <- if ("risk_test_adjusted" %in% names(d)) d$risk_test_adjusted else NA_real_
    raw <- if ("risk_test_raw" %in% names(d)) d$risk_test_raw else NA_real_
    adj_pct <- if ("risk_test_adjusted_pct" %in% names(d)) d$risk_test_adjusted_pct else NA_real_
    raw_pct <- if ("risk_test_raw_pct" %in% names(d)) d$risk_test_raw_pct else NA_real_
    tibble(
      model = model,
      y = d$y_true,
      risk_test = d$risk_test,
      risk_test_adjusted = adj,
      risk_test_adjusted_pct = adj_pct,
      risk_test_raw = raw,
      risk_test_raw_pct = raw_pct,
      run = basename(rd)
    )
  })
  bind_rows(preds_list)
}

find_subgroup_metrics_file <- function(run_dirs, scen, model, set = "test") {
  target <- sprintf("%s__%s__%s_subgroup_metrics.csv", scen, model, set)
  for (rd in run_dirs) {
    f <- file.path(rd, "reports", "subgroups", target)
    if (file.exists(f)) return(f)
  }
  NA_character_
}

#-----------------------------#
# CLI
#-----------------------------#
opt_list <- list(
  make_option(c("--results_root"), type = "character", default = NULL,
              help = "Folder that contains COMBINED/ plus per-model run dirs."),
  make_option(c("--outdir"), type = "character", default = NULL,
              help = "Output directory for figures."),
  make_option(c("--top_features"), type = "integer", default = 25,
              help = "Top N proteins (placeholder for stability plots)."),
  make_option(c("--dca_max_pt"), type = "double", default = 0.05,
              help = "Max threshold for DCA (default 0.05 for rare outcomes)."),
  make_option(c("--dca_step"), type = "double", default = 0.0025),
  make_option(c("--calib_bins"), type = "integer", default = 10,
              help = "Number of bins for calibration/ECE."),
  make_option(c("--spec_targets"), type = "character", default = "0.95,0.99,0.995",
              help = "Comma-separated specificity targets for high-spec comparison.")
)
opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$results_root)) stop("Provide --results_root", call. = FALSE)

results_root <- normalizePath(opt$results_root, mustWork = FALSE)
if (!dir.exists(results_root)) stop("results_root does not exist: ", results_root, call. = FALSE)
if (is.null(opt$outdir)) opt$outdir <- file.path(results_root, "compare_figs")
outdir <- dir_ok(opt$outdir)

# Parse spec targets
spec_targets <- as.numeric(strsplit(opt$spec_targets, ",")[[1]])
spec_targets <- spec_targets[is.finite(spec_targets) & spec_targets > 0 & spec_targets < 1]
if (length(spec_targets) == 0) spec_targets <- c(0.95, 0.99, 0.995)

#-----------------------------#
# Load COMBINED tables
#-----------------------------#
combined <- file.path(results_root, "COMBINED")
paths <- list(
  final = file.path(combined, "core", "final_summary_with_test.csv"),
  test  = file.path(combined, "core", "ALL_test_metrics.csv"),
  cvrep = file.path(combined, "cv",   "ALL_cv_repeat_metrics.csv")
)

final_tbl <- read_csv_or_warn(paths$final, "COMBINED core/final_summary_with_test.csv")
test_tbl  <- read_csv_or_warn(paths$test,  "COMBINED core/ALL_test_metrics.csv")
cvrep_tbl <- read_csv_or_warn(paths$cvrep, "COMBINED cv/ALL_cv_repeat_metrics.csv")
if (is.null(final_tbl) || is.null(test_tbl) || is.null(cvrep_tbl)) {
  stop("Missing COMBINED tables. Run postprocess_compare.py first.", call. = FALSE)
}

if ("repeat" %in% names(cvrep_tbl)) cvrep_tbl <- rename(cvrep_tbl, rep = `repeat`)
if (!"rep" %in% names(cvrep_tbl)) stop("cvrep_tbl missing rep column.", call. = FALSE)

scenarios <- sort(unique(final_tbl$scenario))
models_by_scen <- final_tbl %>% distinct(scenario, model) %>% arrange(scenario, model)

run_dirs <- discover_run_dirs(results_root)
msg("Found %d per-model run dirs under %s", length(run_dirs), results_root)
write_lines(paste0("Results root: ", results_root), file.path(outdir, "RUN_ROOT.txt"))

#============================================================
# OOF-based model selection (NO test leakage)
#============================================================
summarize_oof <- function(scen) {
  df <- cvrep_tbl %>% filter(scenario == scen)

  sum_tbl <- df %>%
    group_by(model) %>%
    summarize(
      n_rep = dplyr::n(),
      AUROC_oof_mean = mean(AUROC_oof, na.rm = TRUE),
      AUROC_oof_sd   = sd(AUROC_oof, na.rm = TRUE),
      PR_AUC_oof_mean = mean(PR_AUC_oof, na.rm = TRUE),
      PR_AUC_oof_sd   = sd(PR_AUC_oof, na.rm = TRUE),
      Brier_oof_mean = mean(Brier_oof, na.rm = TRUE),
      Brier_oof_sd   = sd(Brier_oof, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(PR_AUC_oof_mean), Brier_oof_mean, desc(AUROC_oof_mean)) %>%
    mutate(
      rank_PR = rank(-PR_AUC_oof_mean, ties.method = "min"),
      rank_Brier = rank(Brier_oof_mean, ties.method = "min"),
      rank_AUROC = rank(-AUROC_oof_mean, ties.method = "min"),
      selection_score = paste0("PRrank=", rank_PR, "; Brierrank=", rank_Brier, "; AUROCrank=", rank_AUROC)
    )

  winner <- sum_tbl %>% slice(1) %>% pull(model)
  list(summary = sum_tbl, winner = winner)
}

write_oof_selection <- function(scen) {
  res <- summarize_oof(scen)
  sum_tbl <- res$summary
  winner  <- res$winner

  write_csv(sum_tbl, file.path(outdir, paste0(scen, "__OOF_selection_summary.csv")))
  write_lines(
    c(paste0("Scenario: ", scen),
      paste0("Winner (OOF selection, no test leakage): ", winner),
      "Rule: maximize PR_AUC_oof_mean; tie-break min Brier_oof_mean; then max AUROC_oof_mean",
      "",
      "This selection was made using ONLY out-of-fold predictions on TRAIN data.",
      "TEST metrics are reported for transparency but were NOT used for selection."),
    file.path(outdir, paste0(scen, "__OOF_WINNER.txt"))
  )
  msg("OOF winner for %s: %s", scen, winner)
  invisible(res)
}

#============================================================
# High-specificity comparison (deployment-aligned)
#  - threshold set on TRAIN controls_oof_mean
#  - evaluation on TEST test_preds.csv
#============================================================
compare_highspec_thr_from_trainctrl_eval_on_test <- function(scen, spec_targets = c(0.95, 0.99, 0.995)) {
  msg("High-spec comparison for %s (thr from TRAIN controls_oof; eval on TEST preds)", scen)

  models <- models_by_scen %>% filter(scenario == scen) %>% pull(model)
  out_rows <- list()

  for (mod in models) {

    # TRAIN controls only
    ctrl_file <- find_controls_oof_file(run_dirs, scen, mod)
    if (is.na(ctrl_file) || !file.exists(ctrl_file)) {
      wmsg("Missing controls_oof for %s/%s", scen, mod)
      next
    }
    ctrl_df <- read_csv_or_warn(ctrl_file, sprintf("controls_oof (%s/%s)", scen, mod))
    if (is.null(ctrl_df)) next

    risk_col <- names(ctrl_df)[str_detect(names(ctrl_df), "^risk_.*_oof_mean$")]
    if (length(risk_col) == 0) {
      wmsg("No risk_*_oof_mean column in %s", ctrl_file)
      next
    }
    risk_col <- risk_col[1]
    p_ctrl <- as.numeric(ctrl_df[[risk_col]])
    p_ctrl <- p_ctrl[is.finite(p_ctrl)]
    if (length(p_ctrl) < 20) {
      wmsg("Too few control preds in %s (%d)", ctrl_file, length(p_ctrl))
      next
    }

    # TEST preds (cases + controls)
    test_file <- find_test_pred_file(run_dirs, scen, mod)
    if (is.na(test_file) || !file.exists(test_file)) {
      wmsg("Missing test_preds for %s/%s", scen, mod)
      next
    }
    test_df <- read_csv_or_warn(test_file, sprintf("test_preds (%s/%s)", scen, mod))
    if (is.null(test_df) || !all(c("y_true", "risk_test") %in% names(test_df))) {
      wmsg("Bad/missing columns in %s (need y_true, risk_test)", test_file)
      next
    }
    y_test <- as.integer(test_df$y_true)
    p_test <- as.numeric(test_df$risk_test)

    for (target_spec in spec_targets) {
      thr <- threshold_for_spec(p_ctrl, target_spec)
      if (!is.finite(thr)) {
        out_rows[[length(out_rows) + 1]] <- tibble(
          scenario = scen, model = mod, target_spec = target_spec,
          thr_trainctrl = NA_real_, achieved_spec_trainctrl = NA_real_,
          sens_test = NA_real_, spec_test = NA_real_, ppv_test = NA_real_, npv_test = NA_real_,
          tp_test = NA_integer_, fp_test = NA_integer_, tn_test = NA_integer_, fn_test = NA_integer_,
          n_ctrl_train = length(p_ctrl), n_test = length(y_test), n_test_pos = sum(y_test == 1, na.rm=TRUE),
          FP_per_10k_controls_test = NA_real_, TP_per_10k_people_test = NA_real_,
          note = "could not compute threshold"
        )
        next
      }

      achieved_spec_trainctrl <- mean(p_ctrl < thr)
      mtest <- metrics_at_threshold(y_test, p_test, thr)

      n_test <- mtest$n
      n_pos  <- sum(y_test == 1, na.rm = TRUE)
      n_ctrl_test <- max(1, n_test - n_pos)

      FP_per_10k_controls_test <- if (is.finite(mtest$fp)) (mtest$fp / n_ctrl_test) * 10000 else NA_real_
      TP_per_10k_people_test   <- if (is.finite(mtest$tp)) (mtest$tp / n_test) * 10000 else NA_real_

      out_rows[[length(out_rows) + 1]] <- tibble(
        scenario = scen,
        model = mod,
        target_spec = target_spec,
        thr_trainctrl = thr,
        achieved_spec_trainctrl = achieved_spec_trainctrl,
        sens_test = mtest$sens,
        spec_test = mtest$spec,
        ppv_test  = mtest$ppv,
        npv_test  = mtest$npv,
        tp_test = mtest$tp,
        fp_test = mtest$fp,
        tn_test = mtest$tn,
        fn_test = mtest$fn,
        n_ctrl_train = length(p_ctrl),
        n_test = n_test,
        n_test_pos = n_pos,
        FP_per_10k_controls_test = FP_per_10k_controls_test,
        TP_per_10k_people_test   = TP_per_10k_people_test,
        note = "thr from TRAIN controls_oof_mean; metrics computed on TEST test_preds.csv"
      )
    }
  }

  if (length(out_rows) == 0) {
    wmsg("No high-spec results produced for %s", scen)
    return(NULL)
  }

  df <- bind_rows(out_rows)
  write_csv(df, file.path(outdir, paste0(scen, "__HighSpec_thrFromTrainCtrl__TESTeval.csv")))

  rank_df <- df %>%
    group_by(target_spec) %>%
    mutate(
      rank_sens = rank(-sens_test, ties.method = "min"),
      rank_ppv  = rank(-ppv_test, ties.method = "min")
    ) %>%
    ungroup()

  write_csv(rank_df, file.path(outdir, paste0(scen, "__HighSpec_thrFromTrainCtrl__TESTeval__ranked.csv")))
  msg("Wrote high-spec thr-from-train-controls TEST evaluation for %s", scen)

  df
}

plot_highspec_thr_trainctrl_test_eval <- function(scen, df, winner) {
  if (is.null(df) || nrow(df) == 0) return(invisible(NULL))

  d <- df %>%
    filter(is.finite(sens_test), is.finite(ppv_test)) %>%
    mutate(
      is_winner = (model == winner),
      spec_label = paste0("TrainCtrl thr @ Spec ", sprintf("%.1f%%", target_spec * 100))
    )

  if (nrow(d) == 0) return(invisible(NULL))

  p1 <- ggplot(d, aes(x = reorder(model, sens_test), y = sens_test, fill = is_winner)) +
    geom_col() +
    facet_wrap(~spec_label, scales = "free_y") +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_fill_manual(values = c(`TRUE`="steelblue", `FALSE`="grey60"), guide = "none") +
    theme_bw(base_size = 12) +
    labs(
      title = paste0("Sensitivity at high specificity (TEST) - ", scen),
      subtitle = paste0("Threshold set on TRAIN controls_oof | OOF winner: ", winner),
      x = NULL, y = "Sensitivity (TEST)"
    )
  save_plot(p1, file.path(outdir, paste0(scen, "__HighSpec_thrFromTrainCtrl__sens_TEST.png")), w=10, h=5)

  p2 <- ggplot(d, aes(x = reorder(model, ppv_test), y = ppv_test, fill = is_winner)) +
    geom_col() +
    facet_wrap(~spec_label, scales = "free_y") +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_fill_manual(values = c(`TRUE`="darkgreen", `FALSE`="grey60"), guide = "none") +
    theme_bw(base_size = 12) +
    labs(
      title = paste0("PPV at high specificity (TEST) - ", scen),
      subtitle = paste0("Threshold set on TRAIN controls_oof | OOF winner: ", winner),
      x = NULL, y = "PPV (TEST)"
    )
  save_plot(p2, file.path(outdir, paste0(scen, "__HighSpec_thrFromTrainCtrl__ppv_TEST.png")), w=10, h=5)

  p3 <- ggplot(d, aes(x = FP_per_10k_controls_test, y = TP_per_10k_people_test, color = model, shape = is_winner)) +
    geom_point(size = 3) +
    facet_wrap(~spec_label, scales = "free") +
    scale_shape_manual(values = c(`TRUE`=17, `FALSE`=16), guide = "none") +
    theme_bw(base_size = 12) +
    labs(
      title = paste0("Workload vs yield (TEST) - ", scen),
      subtitle = "X = FP per 10k controls, Y = TP per 10k people | thr from TRAIN controls_oof",
      x = "FP / 10,000 controls (TEST)", y = "TP / 10,000 people (TEST)", color = "Model"
    )
  save_plot(p3, file.path(outdir, paste0(scen, "__HighSpec_thrFromTrainCtrl__workload_vs_yield_TEST.png")), w=11, h=5)

  invisible(NULL)
}

#============================================================
# Combined selection summary (OOF PR-AUC + deployment-aligned view)
#============================================================
write_combined_selection_summary <- function(scen, oof_res, spec_results) {
  primary_winner <- oof_res$winner
  primary_summary <- oof_res$summary

  spec99_best <- NA_character_
  spec99_sens <- NA_real_

  if (!is.null(spec_results) && nrow(spec_results) > 0) {
    d99 <- spec_results %>%
      filter(abs(target_spec - 0.99) < 1e-6, is.finite(sens_test)) %>%
      arrange(desc(sens_test), desc(ppv_test))
    if (nrow(d99) > 0) {
      spec99_best <- d99$model[1]
      spec99_sens <- d99$sens_test[1]
    }
  }

  summary_lines <- c(
    paste0("=" , paste(rep("=", 60), collapse = "")),
    paste0("COMBINED MODEL SELECTION SUMMARY - ", scen),
    paste0("=", paste(rep("=", 60), collapse = "")),
    "",
    "PRIMARY SELECTION (TRAIN OOF PR-AUC - recommended for overall ranking):",
    paste0("  Winner: ", primary_winner),
    paste0("  PR-AUC (OOF mean): ", sprintf("%.4f", primary_summary$PR_AUC_oof_mean[1])),
    "  Rule: max PR-AUC; tie-break min Brier; then max AUROC",
    "",
    "DEPLOYMENT-ALIGNED VIEW (Sensitivity at 99% specificity on TEST):",
    "  (Threshold is computed from TRAIN controls_oof_mean to target 99% specificity on controls.)",
    paste0("  Best model (by Sens@Spec99, TEST): ", ifelse(is.na(spec99_best), "N/A", spec99_best)),
    paste0("  Sens@Spec99 (TEST): ", ifelse(is.finite(spec99_sens), sprintf("%.1f%%", spec99_sens * 100), "N/A")),
    "",
    "INTERPRETATION:",
    if (!is.na(spec99_best) && spec99_best == primary_winner) {
      "  ✓ Primary winner is ALSO best at the high-specificity operating point."
    } else if (!is.na(spec99_best)) {
      c(
        paste0("  ⚠ Primary winner (", primary_winner, ") differs from Spec99 best (", spec99_best, ")."),
        "  Consider deployment context:",
        "    - For general risk stratification / ranking: use primary winner",
        "    - For very high-specificity screening: consider Spec99 best"
      )
    } else {
      "  (High-specificity thresholded evaluation not available.)"
    },
    "",
    "NOTE: Primary selection uses TRAIN OOF only. High-spec view uses TRAIN-derived threshold + TEST evaluation."
  )

  write_lines(summary_lines, file.path(outdir, paste0(scen, "__COMBINED_SELECTION.txt")))
  msg("Wrote combined selection summary for %s", scen)
}

#============================================================
# Paired deltas vs winner (OOF per repeat) with CIs
#============================================================
plot_oof_paired_deltas <- function(scen, winner) {
  df <- cvrep_tbl %>% filter(scenario == scen)

  base <- df %>%
    filter(model == winner) %>%
    select(any_of(c("split_id", "rep")), AUROC_oof, PR_AUC_oof, Brier_oof) %>%
    rename_with(~paste0(.x, "_win"), c(AUROC_oof, PR_AUC_oof, Brier_oof))

  join_cols <- intersect(c("split_id", "rep"), names(df))
  d <- df %>%
    inner_join(base, by = join_cols) %>%
    mutate(
      dAUROC = AUROC_oof - AUROC_oof_win,
      dPR    = PR_AUC_oof - PR_AUC_oof_win,
      dBrier = Brier_oof - Brier_oof_win
    )

  long <- d %>%
    select(model, any_of(c("split_id", "rep")), dAUROC, dPR, dBrier) %>%
    pivot_longer(cols = c(dAUROC, dPR, dBrier), names_to = "metric", values_to = "delta") %>%
    mutate(metric = recode(metric,
                           dAUROC = "Delta AUROC (vs winner)",
                           dPR    = "Delta PR-AUC (vs winner)",
                           dBrier = "Delta Brier (vs winner; + is worse)"))

  sum_tbl <- long %>%
    group_by(model, metric) %>%
    summarize(
      mean_delta = mean(delta, na.rm = TRUE),
      sd_delta   = sd(delta, na.rm = TRUE),
      median_delta = median(delta, na.rm = TRUE),
      CI_lo = quantile(delta, 0.025, na.rm = TRUE),
      CI_hi = quantile(delta, 0.975, na.rm = TRUE),
      n = dplyr::n(),
      .groups = "drop"
    ) %>%
    mutate(
      sig_vs_winner = (CI_lo > 0) | (CI_hi < 0),
      CI_95 = sprintf("[%.4f, %.4f]", CI_lo, CI_hi)
    ) %>%
    arrange(metric, mean_delta)

  write_csv(sum_tbl, file.path(outdir, paste0(scen, "__OOF_paired_deltas_vs_", winner, ".csv")))

  p <- ggplot(long, aes(x = model, y = delta)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", alpha = 0.7) +
    geom_boxplot(width = 0.4, outlier.shape = NA, alpha = 0.5) +
    geom_point(position = position_jitter(width = 0.1), size = 2, alpha = 0.7) +
    facet_wrap(~metric, scales = "free_y") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1)) +
    labs(title = paste0("OOF paired deltas vs selected winner - ", scen),
         subtitle = paste0("Winner: ", winner, " | Red dashed = no difference | TRAIN OOF only"),
         x = NULL, y = "Delta (model - winner)")

  save_plot(p, file.path(outdir, paste0(scen, "__OOF_paired_deltas.png")), w = 12, h = 5)
}

#============================================================
# CV repeat stability plot
#============================================================
plot_cv_repeat_stability <- function(scen, winner) {
  df_order <- cvrep_tbl %>%
    filter(scenario == scen) %>%
    group_by(model) %>%
    summarize(PR = mean(PR_AUC_oof, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(PR)) %>%
    pull(model)

  long <- cvrep_tbl %>%
    filter(scenario == scen) %>%
    mutate(
      model = factor(model, levels = df_order),
      is_winner = (as.character(model) == winner),
      model_label = if_else(is_winner, paste0(as.character(model), " [WINNER]"), as.character(model))
    ) %>%
    select(model_label, rep, AUROC_oof, PR_AUC_oof, Brier_oof) %>%
    pivot_longer(cols = c(AUROC_oof, PR_AUC_oof, Brier_oof),
                 names_to = "metric", values_to = "value") %>%
    mutate(metric = recode(metric,
                           AUROC_oof = "AUROC (OOF)",
                           PR_AUC_oof = "PR-AUC (OOF)",
                           Brier_oof = "Brier (OOF; lower=better)"))

  p <- ggplot(long, aes(x = model_label, y = value)) +
    geom_violin(trim = FALSE, alpha = 0.3, fill = "steelblue") +
    geom_boxplot(width = 0.15, outlier.shape = NA) +
    geom_point(aes(shape = factor(rep)),
               position = position_jitter(width = 0.08),
               size = 2, alpha = 0.8) +
    facet_wrap(~metric, scales = "free_y") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1)) +
    labs(title = paste0("TRAIN CV stability across repeats - ", scen),
         subtitle = paste0("OOF-selected winner: ", winner),
         x = NULL, y = NULL, shape = "Repeat")

  save_plot(p, file.path(outdir, paste0(scen, "__CV_repeat_stability.png")), w = 12, h = 5)
}

#============================================================
# TEST forest plot (report only - not used for selection)
#============================================================
plot_test_forest <- function(scen, winner) {
  df_raw <- final_tbl %>% filter(scenario == scen)

  df <- df_raw %>%
    group_by(model) %>%
    summarize(
      AUROC_test = mean(AUROC_test, na.rm = TRUE),
      PR_AUC_test = mean(PR_AUC_test, na.rm = TRUE),
      Brier_test = mean(Brier_test, na.rm = TRUE),
      AUROC_test_95CI = first(na.omit(AUROC_test_95CI)),
      PR_AUC_test_95CI = first(na.omit(PR_AUC_test_95CI)),
      Brier_test_95CI = first(na.omit(Brier_test_95CI)),
      .groups = "drop"
    ) %>%
    mutate(
      panel_n = parse_panel_n(model),
      is_panel = !is.na(panel_n),
      is_winner = (model == winner)
    ) %>%
    arrange(desc(PR_AUC_test)) %>%
    mutate(
      model_label = if_else(is_winner, paste0(model, " [OOF WINNER]"), model),
      model_label = factor(model_label, levels = rev(unique(model_label)))
    )

  long <- df %>%
    select(model_label, AUROC_test, PR_AUC_test, Brier_test,
           AUROC_test_95CI, PR_AUC_test_95CI, Brier_test_95CI) %>%
    pivot_longer(cols = c(AUROC_test, PR_AUC_test, Brier_test),
                 names_to = "metric", values_to = "value") %>%
    mutate(
      metric = recode(metric,
                      AUROC_test = "AUROC (TEST)",
                      PR_AUC_test = "PR-AUC (TEST)",
                      Brier_test = "Brier (TEST; lower=better)"),
      ci_str = case_when(
        metric == "AUROC (TEST)" ~ AUROC_test_95CI,
        metric == "PR-AUC (TEST)" ~ PR_AUC_test_95CI,
        metric == "Brier (TEST; lower=better)" ~ Brier_test_95CI
      )
    ) %>%
    bind_cols(parse_ci_string(.$ci_str)) %>%
    mutate(has_ci = is.finite(lo) & is.finite(hi))

  p <- ggplot(long, aes(x = value, y = model_label)) +
    geom_point(size = 2.5) +
    geom_errorbarh(data = filter(long, has_ci),
                   aes(xmin = lo, xmax = hi),
                   height = 0.2, linewidth = 0.6) +
    facet_wrap(~metric, scales = "free_x") +
    theme_bw(base_size = 12) +
    labs(title = paste0("TEST performance (for reporting) - ", scen),
         subtitle = paste0("OOF-selected winner: ", winner, " | TEST was NOT used for selection"),
         x = NULL, y = NULL)

  save_plot(p, file.path(outdir, paste0(scen, "__TEST_forest.png")), w = 11, h = 5)

  rank_tbl <- df %>%
    mutate(
      rank_PR = rank(-PR_AUC_test, ties.method = "min"),
      rank_AUROC = rank(-AUROC_test, ties.method = "min"),
      rank_Brier = rank(Brier_test, ties.method = "min")
    ) %>%
    select(model, is_winner, AUROC_test, PR_AUC_test, Brier_test,
           rank_AUROC, rank_PR, rank_Brier) %>%
    arrange(rank_PR)

  write_csv(rank_tbl, file.path(outdir, paste0(scen, "__TEST_metric_ranks.csv")))
}

#============================================================
# TEST curves (PR, ROC, Calibration, DCA)
#============================================================
plot_test_curves_all <- function(scen, winner, dca_max_pt = 1.0, dca_step = 0.001, calib_bins = 10) {
  dfm <- models_by_scen %>% filter(scenario == scen)

  preds_list <- map(dfm$model, function(mod) {
    f <- find_test_pred_file(run_dirs, scen, mod)
    if (is.na(f) || !file.exists(f)) return(NULL)
    d <- read_csv_or_warn(f, sprintf("test preds (%s/%s)", scen, mod))
    if (is.null(d) || !all(c("y_true", "risk_test") %in% names(d))) return(NULL)
    tibble(model = mod, y = d$y_true, p = d$risk_test)
  })
  preds <- bind_rows(preds_list)

  if (nrow(preds) == 0) {
    wmsg("No test preds found for %s", scen)
    return(invisible(NULL))
  }

  prev <- mean(preds$y == 1, na.rm = TRUE)

  # PR curves
  pr_df <- preds %>%
    group_by(model) %>%
    group_modify(~pr_curve(.x$y, .x$p)) %>%
    ungroup() %>%
    mutate(is_winner = (model == winner))

  p_pr <- ggplot(pr_df, aes(x = recall, y = precision, color = model, linewidth = is_winner)) +
    geom_line() +
    scale_linewidth_manual(values = c(`TRUE` = 1.5, `FALSE` = 0.8), guide = "none") +
    geom_hline(yintercept = prev, linetype = "dashed", alpha = 0.6) +
    theme_bw(base_size = 12) +
    theme(plot.margin = margin(t = 2, r = 1, b = 1, l = 1, unit = "cm")) +
    labs(title = paste0("PR curves (TEST) - ", scen),
         subtitle = sprintf("Dashed = prevalence (%.4f) | Winner: %s", prev, winner),
         x = "Recall (Sensitivity)", y = "Precision (PPV)", color = "Model")
  save_plot(p_pr, file.path(outdir, paste0(scen, "__TEST_PR_curves.png")), w = 8.5, h = 6)

  # ROC curves
  roc_df <- preds %>%
    group_by(model) %>%
    group_modify(~roc_curve(.x$y, .x$p)) %>%
    ungroup() %>%
    mutate(is_winner = (model == winner))

  p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = model, linewidth = is_winner)) +
    geom_line() +
    scale_linewidth_manual(values = c(`TRUE` = 1.5, `FALSE` = 0.8), guide = "none") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.6) +
    coord_equal() +
    theme_bw(base_size = 12) +
    theme(plot.margin = margin(t = 2, r = 1, b = 1, l = 1, unit = "cm")) +
    labs(title = paste0("ROC curves (TEST) - ", scen),
         subtitle = paste0("Winner: ", winner),
         x = "False Positive Rate (1 - Specificity)", y = "True Positive Rate (Sensitivity)",
         color = "Model")
  save_plot(p_roc, file.path(outdir, paste0(scen, "__TEST_ROC_curves.png")), w = 7.5, h = 6)

  # Calibration curves + summary
  cal_df <- preds %>%
    group_by(model) %>%
    group_modify(~calibration_bins(.x$y, .x$p, n_bins = calib_bins)) %>%
    ungroup()

  cal_summary <- preds %>%
    group_by(model) %>%
    group_modify(~{
      cal <- calibration_bins(.x$y, .x$p, n_bins = calib_bins)
      tibble(
        n_test = length(.x$y),
        prevalence = mean(.x$y == 1),
        ECE = ece_from_bins(cal),
        Cal_RMSE = cal_rmse_from_bins(cal)
      )
    }) %>%
    ungroup()

  calib_params <- test_tbl %>%
    filter(scenario == scen) %>%
    select(model, calibration_intercept_test, calibration_slope_test) %>%
    mutate(
      calib_quality = purrr::map2_chr(calibration_intercept_test, calibration_slope_test, assess_calibration)
    )

  cal_summary <- cal_summary %>%
    left_join(calib_params, by = "model") %>%
    mutate(is_winner = (model == winner)) %>%
    arrange(ECE)

  write_csv(cal_summary, file.path(outdir, paste0(scen, "__TEST_calibration_summary.csv")))

  if (nrow(cal_df) > 0) {
    p_cal <- ggplot(cal_df, aes(x = prob_pred, y = prob_true, color = model)) +
      geom_line(linewidth = 1) +
      geom_point(size = 2.5, alpha = 0.8) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.6) +
      theme_bw(base_size = 12) +
      theme(plot.margin = margin(t = 2, r = 1, b = 1, l = 1, unit = "cm")) +
      labs(title = paste0("Calibration (TEST) - ", scen),
           subtitle = paste0("Dashed = perfect calibration | Winner: ", winner,
                             " | See *_calibration_summary.csv for ECE/RMSE"),
           x = "Mean predicted probability", y = "Observed event rate", color = "Model")
    save_plot(p_cal, file.path(outdir, paste0(scen, "__TEST_calibration.png")), w = 8.5, h = 6)
  }

  # DCA
  pts <- seq(0.0005, dca_max_pt, by = dca_step)

  dca_all <- preds %>%
    group_by(model) %>%
    group_modify(~{
      yv <- .x$y
      pv <- .x$p
      tibble(
        threshold = pts,
        net_benefit = map_dbl(pts, function(pt) net_benefit(yv, pv, pt))
      )
    }) %>%
    ungroup() %>%
    mutate(is_winner = (model == winner))

  dca_ref <- bind_rows(
    tibble(threshold = pts, net_benefit = 0, strategy = "Treat none"),
    tibble(threshold = pts,
           net_benefit = prev - (1 - prev) * (pts / (1 - pts)),
           strategy = "Treat all")
  )

  p_dca <- ggplot() +
    geom_line(data = dca_ref,
              aes(x = threshold, y = net_benefit, linetype = strategy),
              color = "grey40", linewidth = 1) +
    geom_line(data = dca_all,
              aes(x = threshold, y = net_benefit, color = model, linewidth = is_winner)) +
    scale_linewidth_manual(values = c(`TRUE` = 1.5, `FALSE` = 0.8), guide = "none") +
    theme_bw(base_size = 12) +
    labs(title = paste0("Decision Curve Analysis (TEST) - ", scen),
         subtitle = sprintf("Threshold range: %.4f to %.2f | Winner: %s", dca_step, dca_max_pt, winner),
         x = "Threshold probability", y = "Net benefit",
         color = "Model", linetype = NULL)
  save_plot(p_dca, file.path(outdir, paste0(scen, "__TEST_DCA.png")), w = 9.5, h = 6)
}

#============================================================
# TEST risk score distributions (all runs combined)
#============================================================
plot_test_risk_score_distributions <- function(scen) {
  models <- models_by_scen %>% filter(scenario == scen) %>% pull(model)
  preds_list <- map(models, function(mod) {
    d <- collect_test_preds_all_runs(run_dirs, scen, mod)
    if (is.null(d) || nrow(d) == 0) {
      wmsg("No test preds across runs for %s/%s", scen, mod)
      return(NULL)
    }
    d
  })
  preds <- bind_rows(preds_list)

  if (nrow(preds) == 0) {
    wmsg("No test preds found for risk distributions: %s", scen)
    return(invisible(NULL))
  }

  preds <- preds %>%
    mutate(
      outcome = if_else(as.integer(y) == 1, "Incident CeD", "Control"),
      model = factor(model, levels = models)
    )

  percent_accuracy_for_max <- function(max_prob) {
    max_pct <- max_prob * 100
    if (!is.finite(max_pct) || max_pct <= 0) return(1)
    if (max_pct < 0.1) return(0.001)
    if (max_pct < 1) return(0.01)
    if (max_pct < 5) return(0.1)
    1
  }

  number_accuracy_for_max <- function(max_prob) {
    if (!is.finite(max_prob) || max_prob <= 0) return(0.01)
    if (max_prob < 1e-4) return(1e-6)
    if (max_prob < 1e-3) return(1e-5)
    if (max_prob < 1e-2) return(1e-4)
    if (max_prob < 1e-1) return(1e-3)
    0.01
  }

  plot_density <- function(df, score_col, title, subtitle, file_suffix,
                           x_limits = c(0, 1), x_labels = percent_format(accuracy = 1),
                           x_title = "Predicted risk") {
    df <- df %>% filter(is.finite(.data[[score_col]]))
    if (nrow(df) == 0) {
      wmsg("No finite %s values for %s", score_col, scen)
      return(invisible(NULL))
    }
    p <- ggplot(df, aes(x = .data[[score_col]], fill = outcome, color = outcome)) +
      geom_density(alpha = 0.35, adjust = 1.1, linewidth = 0.6) +
      facet_wrap(~model, ncol = 2, scales = "free_y") +
      scale_x_continuous(limits = x_limits, labels = x_labels) +
      theme_bw(base_size = 12) +
      labs(title = title,
           subtitle = subtitle,
           x = x_title, y = "Density", fill = "Group", color = "Group")

    save_plot(p, file.path(outdir, paste0(scen, "__TEST_risk_score_distributions", file_suffix, ".png")), w = 10, h = 6)
  }

  plot_density(
    preds,
    "risk_test",
    paste0("Risk score distributions (TEST, all runs) - ", scen),
    "risk_test; combined across seeds/runs; controls vs incident cases",
    ""
  )

  if ("risk_test_adjusted" %in% names(preds) && any(is.finite(preds$risk_test_adjusted))) {
    adj_max <- max(preds$risk_test_adjusted, na.rm = TRUE)
    adj_lim <- if (is.finite(adj_max) && adj_max > 0) c(0, min(adj_max * 1.05, 1)) else c(0, 1)
    adj_acc <- number_accuracy_for_max(adj_max)
    plot_density(
      preds,
      "risk_test_adjusted",
      paste0("Risk score distributions (TEST, adjusted, all runs) - ", scen),
      "risk_test_adjusted (prevalence-shifted); x-axis zoomed to adjusted range",
      "_adjusted",
      x_limits = adj_lim,
      x_labels = label_number(accuracy = adj_acc),
      x_title = "Predicted risk (probability)"
    )
  }

  if ("risk_test_raw" %in% names(preds) && any(is.finite(preds$risk_test_raw))) {
    plot_density(
      preds,
      "risk_test_raw",
      paste0("Risk score distributions (TEST, raw, all runs) - ", scen),
      "risk_test_raw (unadjusted); combined across seeds/runs",
      "_raw"
    )
  }
}

#============================================================
# High-specificity operating points (from ALL_test_metrics.csv)
#============================================================
plot_high_spec_operating_points <- function(scen, winner) {
  df <- test_tbl %>% filter(scenario == scen)

  prec_cols <- names(df)[str_detect(names(df), "^precision_test_at_spec\\d+_ctrl$")]
  rec_cols  <- names(df)[str_detect(names(df), "^recall_test_at_spec\\d+_ctrl$")]
  fp_cols   <- names(df)[str_detect(names(df), "^fp_test_at_spec\\d+_ctrl$")]

  if (length(prec_cols) == 0) {
    wmsg("No high-spec columns for %s", scen)
    return(invisible(NULL))
  }

  long <- df %>%
    select(model, n_test, n_test_pos, all_of(prec_cols), all_of(rec_cols), all_of(fp_cols)) %>%
    pivot_longer(cols = -c(model, n_test, n_test_pos),
                 names_to = "key", values_to = "value") %>%
    mutate(spec = str_extract(key, "spec\\d+")) %>%
    group_by(model, spec) %>%
    summarize(
      PPV = value[str_detect(key, "^precision")][1],
      Sens = value[str_detect(key, "^recall")][1],
      FP = value[str_detect(key, "^fp")][1],
      n_test = n_test[1],
      n_test_pos = n_test_pos[1],
      .groups = "drop"
    ) %>%
    mutate(
      n_controls = pmax(1, n_test - n_test_pos),
      FP_per_10k_controls = (FP / n_controls) * 10000,
      TP = round(Sens * n_test_pos),
      TP_per_10k_people = (TP / n_test) * 10000,
      is_winner = (model == winner),
      model_label = if_else(is_winner, paste0(model, " [WINNER]"), model),
      NNS = if_else(PPV > 0, round(1 / PPV), NA_real_)
    )

  p1 <- ggplot(long, aes(x = Sens, y = PPV, shape = is_winner)) +
    geom_point(size = 3) +
    geom_text(aes(label = model), hjust = -0.1, vjust = -0.3, size = 3) +
    facet_wrap(~spec) +
    scale_x_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    scale_shape_manual(values = c(`TRUE` = 17, `FALSE` = 16), guide = "none") +
    theme_bw(base_size = 12) +
    labs(title = paste0("High-specificity operating points (TEST) - ", scen),
         subtitle = paste0("Winner: ", winner),
         x = "Sensitivity (Recall)", y = "PPV (Precision)")
  save_plot(p1, file.path(outdir, paste0(scen, "__TEST_highSpec_PPV_vs_Sens.png")), w = 10.5, h = 5)

  p2 <- ggplot(long, aes(x = reorder(model_label, FP_per_10k_controls), y = FP_per_10k_controls)) +
    geom_col(aes(fill = is_winner)) +
    scale_fill_manual(values = c(`TRUE` = "steelblue", `FALSE` = "grey60"), guide = "none") +
    facet_wrap(~spec, scales = "free_y") +
    coord_flip() +
    theme_bw(base_size = 12) +
    labs(title = paste0("False-positive workload at high specificity - ", scen),
         x = NULL, y = "FP per 10,000 controls")
  save_plot(p2, file.path(outdir, paste0(scen, "__TEST_highSpec_FP_workload.png")), w = 10.5, h = 5)

  p3 <- ggplot(long, aes(x = reorder(model_label, TP_per_10k_people), y = TP_per_10k_people)) +
    geom_col(aes(fill = is_winner)) +
    scale_fill_manual(values = c(`TRUE` = "darkgreen", `FALSE` = "grey60"), guide = "none") +
    facet_wrap(~spec, scales = "free_y") +
    coord_flip() +
    theme_bw(base_size = 12) +
    labs(title = paste0("True-positive yield at high specificity - ", scen),
         subtitle = "Cases detected per 10,000 people screened",
         x = NULL, y = "TP per 10,000 people")
  save_plot(p3, file.path(outdir, paste0(scen, "__TEST_highSpec_TP_yield.png")), w = 10.5, h = 5)

  write_csv(long, file.path(outdir, paste0(scen, "__TEST_highSpec_summary.csv")))
}

#============================================================
# Pairwise bootstrap heatmap (from COMBINED)
#============================================================
plot_pairwise_bootstrap_heatmap <- function(scen) {
  f <- file.path(combined, "core", paste0(scen, "__test_model_comparisons_bootstrap.csv"))
  if (!file.exists(f)) {
    wmsg("No pairwise bootstrap for %s", scen)
    return(invisible(NULL))
  }

  df <- read_csv_or_warn(f, "pairwise bootstrap")
  if (is.null(df) || nrow(df) == 0) return(invisible(NULL))

  df2 <- df %>%
    bind_cols(parse_ci_string(df$diff_95CI)) %>%
    mutate(sig = is.finite(lo) & is.finite(hi) & (lo > 0 | hi < 0))

  for (met in unique(df2$metric)) {
    d <- df2 %>% filter(metric == met)

    p <- ggplot(d, aes(x = model2, y = model1, fill = diff_model1_minus_model2)) +
      geom_tile(color = "grey85") +
      geom_text(aes(label = if_else(sig, "*", "")), size = 6, vjust = 0.8) +
      scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0) +
      theme_bw(base_size = 11) +
      theme(axis.text.x = element_text(angle = 35, hjust = 1)) +
      labs(title = paste0("Pairwise TEST differences - ", scen, " / ", met),
           subtitle = "* = 95% CI excludes 0",
           x = "Model 2", y = "Model 1", fill = "Diff\n(M1 - M2)")
    save_plot(p, file.path(outdir, paste0(scen, "__pairwise_bootstrap__", met, ".png")), w = 10.5, h = 7)
  }
}

#============================================================
# Clinical summary table
#============================================================
write_clinical_summary <- function(scen, winner) {
  test_winner <- test_tbl %>% filter(scenario == scen, model == winner)
  oof_summary <- cvrep_tbl %>%
    filter(scenario == scen, model == winner) %>%
    summarize(
      PR_AUC_oof_mean = mean(PR_AUC_oof, na.rm = TRUE),
      PR_AUC_oof_sd = sd(PR_AUC_oof, na.rm = TRUE),
      AUROC_oof_mean = mean(AUROC_oof, na.rm = TRUE),
      Brier_oof_mean = mean(Brier_oof, na.rm = TRUE)
    )

  get_col <- function(df, col, default = NA) {
    if (col %in% names(df) && nrow(df) > 0) df[[col]][1] else default
  }

  summary_tbl <- tibble(
    scenario = scen,
    selected_model = winner,
    selection_method = "OOF PR-AUC (no test leakage)",

    OOF_PR_AUC = sprintf("%.3f (SD=%.3f)", oof_summary$PR_AUC_oof_mean, oof_summary$PR_AUC_oof_sd),
    OOF_AUROC  = sprintf("%.3f", oof_summary$AUROC_oof_mean),
    OOF_Brier  = sprintf("%.4f", oof_summary$Brier_oof_mean),

    TEST_PR_AUC = sprintf("%.3f %s", get_col(test_winner, "PR_AUC_test"),
                          ifelse(is.na(get_col(test_winner, "PR_AUC_test_95CI")), "",
                                 get_col(test_winner, "PR_AUC_test_95CI"))),
    TEST_AUROC = sprintf("%.3f %s", get_col(test_winner, "AUROC_test"),
                         ifelse(is.na(get_col(test_winner, "AUROC_test_95CI")), "",
                                get_col(test_winner, "AUROC_test_95CI"))),
    TEST_Brier = sprintf("%.4f", get_col(test_winner, "Brier_test")),

    n_train = get_col(test_winner, "n_train"),
    n_train_cases = get_col(test_winner, "n_train_pos"),
    n_test = get_col(test_winner, "n_test"),
    n_test_cases = get_col(test_winner, "n_test_pos"),
    prevalence_test = sprintf("%.2f%%", 100 * get_col(test_winner, "n_test_pos") / get_col(test_winner, "n_test")),

    calibration_intercept = round(get_col(test_winner, "calibration_intercept_test"), 3),
    calibration_slope = round(get_col(test_winner, "calibration_slope_test"), 3),
    calibration_quality = assess_calibration(get_col(test_winner, "calibration_intercept_test"),
                                             get_col(test_winner, "calibration_slope_test"))
  )

  write_csv(summary_tbl, file.path(outdir, paste0(scen, "__CLINICAL_SUMMARY.csv")))
  msg("Wrote clinical summary for %s", scen)
}

#============================================================
# Subgroup calibration / fairness summary
#============================================================
msg("Scanning for subgroup fairness metrics")
subgroup_entries <- list()
for (scen in scenarios) {
  scen_models <- models_by_scen %>% filter(scenario == scen) %>% pull(model)
  for (mod in scen_models) {
    f <- find_subgroup_metrics_file(run_dirs, scen, mod, set = "test")
    if (is.na(f) || !file.exists(f)) next
    tbl <- read_csv_or_warn(f, sprintf("subgroup metrics (%s/%s)", scen, mod))
    if (is.null(tbl) || nrow(tbl) == 0) next
    tbl <- tbl %>%
      mutate(
        scenario = scen,
        model = mod,
        group = factor(group, levels = sort(unique(group)))
      )
    subgroup_entries[[length(subgroup_entries) + 1]] <- tbl
  }
}

if (length(subgroup_entries) > 0) {
  subgroup_tbl <- bind_rows(subgroup_entries)
  write_csv(subgroup_tbl, file.path(outdir, "ALL_subgroup_metrics.csv"))

  p_fair <- subgroup_tbl %>%
    ggplot(aes(x = group, y = AUROC, fill = model)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    facet_wrap(~scenario, scales = "free_x") +
    coord_cartesian(ylim = c(0.4, 1.0)) +
    labs(title = "Subgroup AUROC (test split)", x = "Genetic ethnic grouping", y = "AUROC") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  save_plot(p_fair, file.path(outdir, "subgroup_AUROC.png"), w = 11, h = 6)

  p_cal <- subgroup_tbl %>%
    ggplot(aes(x = group, y = calibration_slope, color = model)) +
    geom_hline(yintercept = 1.0, linetype = "dashed", color = "grey50") +
    geom_point(size = 2.4, position = position_dodge(width = 0.5)) +
    facet_wrap(~scenario, scales = "free_x") +
    coord_cartesian(ylim = c(0.0, 2.0)) +
    labs(title = "Subgroup Calibration Slopes (test split)", x = "Genetic ethnic grouping", y = "Calibration slope") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  save_plot(p_cal, file.path(outdir, "subgroup_calibration_slope.png"), w = 11, h = 6)
} else {
  msg("No subgroup metrics found; skipping fairness plots.")
}

#============================================================
# Main execution
#============================================================
msg("Generating figure set into: %s", outdir)
msg("Specificity targets: %s", paste(spec_targets, collapse = ", "))

oof_selection <- map(scenarios, function(scen) write_oof_selection(scen))
names(oof_selection) <- scenarios

for (scen in scenarios) {
  winner <- oof_selection[[scen]]$winner
  msg("======== Scenario: %s | OOF Winner: %s ========", scen, winner)

  # Deployment-aligned high-specificity evaluation
  spec_results <- compare_highspec_thr_from_trainctrl_eval_on_test(scen, spec_targets = spec_targets)
  plot_highspec_thr_trainctrl_test_eval(scen, spec_results, winner)
  write_combined_selection_summary(scen, oof_selection[[scen]], spec_results)

  # Existing analyses
  plot_oof_paired_deltas(scen, winner)
  plot_cv_repeat_stability(scen, winner)

  plot_test_forest(scen, winner)
  plot_test_curves_all(scen, winner,
                       dca_max_pt = opt$dca_max_pt,
                       dca_step = opt$dca_step,
                       calib_bins = opt$calib_bins)
  plot_test_risk_score_distributions(scen)
  plot_high_spec_operating_points(scen, winner)
  plot_pairwise_bootstrap_heatmap(scen)

  write_clinical_summary(scen, winner)
}

msg("DONE. Outputs in: %s", outdir)
