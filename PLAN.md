Plan: SCS 3546 Assignment 1 Work Plan
-------------------------------------

The notebook builds Keras regression models on a 13-feature housing-price dataset: 404 training rows, 102 test rows, with the target in thousands of dollars. Part 1 compares three model shapes. Part 2 compares four 64/128-unit models using 3-fold cross-validation over 100 epochs. Part 1 uses a single shuffled 80/20 split (`train_test_split`) for 100 epochs. Part 2 uses a manual `KFold` loop that fits the scaler inside each fold and records `history`. That gives you the learning curves and the cross-validated MAE ± std from one function, without needing scikeras.

### 1\. What each part asks for (15 marks total)

| Part | Marks | Spec, exactly as the template words it |
| --- | --- | --- |
| 1a Baseline | 2 | "one single fully-connected hidden layer with the same number of neurons as input features (i.e. 13 neurons)"; standardize features; no activation on the output layer; ADAM optimizer on mean squared error loss; "Plot learning curves and report on both training and validation performance." |
| 1b Deeper | 1 | "2 dense layers having a smaller number of neurons (e.g. 16, 8)" |
| 1c Wider | 1 | "a wider model with more neurons (e.g. 32, 16)" |
| 2a Model 1 | 2 | Dense 64 ReLU → Dense 64 ReLU; "appropriate output layer and activation"; 100 epochs; CV "(e.g. with 3 cross-folds)"; plot loss and MAE for training and validation; "Report MAE from CV with standard deviation." |
| 2b Model 2 | 2 | Dense 128 ReLU → Dense 64 ReLU; everything else as in 2a |
| 2c Model 3 | 2 | "Same as Model 2, but use tanh activation functions instead of relu." |
| 2d Model 4 | 2 | "Same as Model 2, but use the rmsprop optimizer when training." |
| 2e Comparison | 1 | "Which model performed best?" plus why, in terms of hyperparameter theory; "provide visualizations (e.g. tables or comparison plot)" |
| Clarity | 2 | Documented code, readable names, repeated code turned into functions, explanations backed by plots |

These rules apply to every part: MSE is the loss, and you don't need to do EDA. The template states that raw performance matters less than "the changes that you observe when changing model architectures."

### 2\. Checklist of required outputs

-   [ ]  1a: model built (13-unit ReLU hidden layer, `Dense(1)` linear output, Adam, MSE); 80/20 split, 100 epochs; learning-curve plot (training vs validation loss); reported training and validation MSE, with RMSE in $1000s also suggested; markdown write-up.
-   [ ]  1b: 16→8 ReLU model; learning curves; training and validation metrics; markdown comparing it to 1a, noting that 1a→1b changes depth *and* width.
-   [ ]  1c: 32→16 ReLU model; learning curves; training and validation metrics; markdown comparing it to 1a and 1b, noting that 1b→1c changes width only.
-   [ ]  1a--1c (all three): same 80/20 split and same seed, so differences come from the architecture only.
-   [ ]  2a--2d (each one): model as specified; 3-fold CV with 100 epochs; loss curve (training and validation); MAE curve (training and validation); CV MAE mean ± std; markdown write-up.
-   [ ]  2e: summary table with CV MAE mean, CV MAE std and CV MSE for all four models; one comparison plot (bar chart with error bars and/or overlaid validation-MAE curves); markdown answer to "which model was best and why."
-   [ ]  Clarity: helper functions with one-line docstrings; constants at the top; one consistent plot style.
-   [ ]  Optional (after 2e): best model retrained on the full training set, test MAE reported once.

### 3\. Helper functions (put them in one "Helper Functions" section near the top)

-   `set_global_seed` --- seeds Python, NumPy and TensorFlow in one call (via `keras.utils.set_random_seed`).
-   `get_data_dir` --- detects Colab vs local and returns the folder that holds the CSVs.
-   `load_data` --- reads the four CSVs with `index_col=0` and returns NumPy arrays (X: n×13, y: n).
-   `build_model` --- builds and compiles a `Sequential` model from `hidden_units`, `activation` and `optimizer`, with a linear `Dense(1)` output, MSE loss and an MAE metric.
-   `scale_features` --- fits a `StandardScaler` on training rows only and transforms both the training and validation/test rows.
-   `train_with_validation` --- Part 1: splits with `train_test_split(test_size=0.2, random_state=SEED)`, fits the scaler on the 80% only, trains with `validation_data=(X_val, y_val)`, and returns the fitted model and its `history`.
-   `run_cross_validation` --- runs the 3-fold `KFold` loop (fixed `random_state`), scales per fold, builds a fresh model per fold, trains for 100 epochs, and returns per-fold histories and final validation MSE/MAE.
-   `plot_learning_curves` --- plots training vs validation curves for one metric (loss or MAE). For CV it shows the mean across folds with a ±std band.
-   `summarize_results` --- turns per-fold scores into mean ± std (MAE, MSE, RMSE) as a dict or DataFrame row.
-   `plot_model_comparison` --- bar chart of CV MAE with std error bars and/or overlaid validation-MAE curves for 2e.
-   `evaluate_on_test` (optional) --- retrains the chosen model on the full scaled training set and returns its test MSE/MAE.

### 4\. Work order (assuming roughly 1--2 days)

Day 1

1.  Setup (about 1h): local venv and kernel; open in Colab; confirm both environments import TF, sklearn and pandas.
2.  Setup cell and data loading (1h or less, including the quick Colab check): environment and seed cell, `get_data_dir`, `load_data` and `set_global_seed`, and the data-import cell edit. Check shapes are (404, 13) and (102, 13). The remaining helpers are written in steps 3 and 6.
3.  Write `build_model`, `scale_features` and `plot_learning_curves`, then run 1a end to end (about 1.5h).
4.  Run 1b and 1c with the same helpers (about 1h).
5.  Write the Part 1 markdown answers (about 1h).

Day 2

6.  Write `run_cross_validation` and `summarize_results`, then run 2a (about 1.5h).
7.  Run 2b, 2c and 2d, which only change parameters (about 1h; training itself takes minutes).
8.  Part 2e: table, comparison plot and answer (about 1h).
9.  Clarity pass: docstrings, naming, remove dead code (about 0.5h).
10. Final checks: Restart and Run All locally, then in Colab, then save the fully executed notebook (about 0.5--1h).
11. Optional: test MAE for the best model (see decision 10).

Priority: the total is about 11 hours. Do the required parts first; the test-set evaluation and extra polish are optional. If time is short, the 2e comparison table and the markdown write-ups earn more marks than extra plots.

### 5\. Decisions to justify in markdown

1.  Where the scaler is fitted. The template hint says "standardize your data upfront," but fitting on all training data before CV leaks validation-fold statistics. Decision: fit the scaler inside each fold in Part 2, and on the 80% training split only in Part 1. Open that markdown cell with "Deliberate deviation from the template hint:" and explain why in one sentence, so the grader doesn't read it as missing the hint. The imported `Pipeline`/`StandardScaler` points the same way.
2.  Target scaling. Recommendation: leave `y` unscaled, so MAE and RMSE stay in $1000s, which matches the preamble's "thousands of dollars" note.
3.  Same folds and seeds for all models. Use one `KFold(n_splits=3, shuffle=True, random_state=SEED)` and reseed before each model, so differences come from the architecture, not from the split.
4.  Fresh model per fold. Before every model (each CV fold and each of 1a--1c), use this order: `keras.backend.clear_session()`, then `keras.utils.set_random_seed(SEED)`, then build. Clearing after seeding can reset Keras 3's seed generator. Same seed every time means each fold, and models with identical layer sizes (Model 2 vs Model 4), start from the same initial weights, so only the data or the optimizer differs.
5.  Manual KFold vs scikeras `cross_val_score`. `cross_val_score` doesn't return per-epoch history, and the plots need it. Recommendation: manual loop, scikeras not needed.
6.  Which number is the reported CV MAE. Decision: the validation MAE at the final epoch (epoch 100) of each fold, then the mean ± std across the 3 folds. Say this explicitly, and why: picking the "best epoch" quietly uses validation data to choose results. Plot CV curves as the mean across folds with a ±std band, which directly supports the reported ± std.
7.  Output layer. `Dense(1)` with linear activation. Justify it by the regression target having no bounds.
8.  Settings left at defaults. Batch size (Keras default 32) and learning rate (optimizer default) aren't in the spec. Keep them the same for every model and state them. RMSprop and Adam both default to 0.001, so Model 4 changes only the optimizer: a clean one-variable comparison with Model 2.
9.  Input layer. Use `keras.Input(shape=(13,))`. Keras 3 warns about `input_dim`/`input_shape` on `Dense`.
10. When to use the test set. Decision: only after 2e, and only for the best model, retrained on the full training set; report its test MAE once. It must not influence which model is picked.
11. Part 1 hidden activation. Decision: ReLU (the template doesn't name one). State it in one line.
12. 1b layer sizes. Decision: 16/8 as written, even though 16 > 13. In the write-up, point out the confound: 1a→1b changes depth and width, 1b→1c changes width only.
13. Part 1 validation split. Don't use Keras `validation_split=0.2`: it takes the *last* 20% of rows without shuffling, and with upfront scaling that slice would also have influenced the scaler. Decision: `train_test_split(..., test_size=0.2, random_state=SEED)`, scaler fitted on the 80%, `validation_data=(X_val, y_val)`, 100 epochs (same as Part 2).

### 6\. Environment notes

Data path (one setup cell)

-   Detect Colab with `"google.colab" in sys.modules` (or a try-import).
-   In Colab: mount Drive and set the folder to your Drive folder, or to `/content/` if you upload the files directly.
-   Locally: VS Code usually starts the kernel in `notebooks/`, but not always. Try the candidate folders in order (Colab folder, `data`, `../data`) and use the first one that contains `X_train.csv`. If none does, raise a clear error listing the paths tried.
-   Filename case trap: the files are `X_train.csv` / `X_test.csv` (capital X). The template reads `x_train.csv`. Windows ignores case, Colab (Linux) does not, so it fails there. Use the real names.
-   Leading index column: each CSV starts with an unnamed index column. Read with `index_col=0`, or you'll get 14 features.
-   The template's import cell calls `from google.colab import drive` with no check, so it crashes locally. Decision: edit that cell (the template invites you to "update paths below") so `drive.mount` runs only in Colab and the paths use the detected data folder; add a short comment explaining the change.

Package versions

-   Local (requirements.txt): TensorFlow 2.16.2 (ships Keras 3), scikit-learn 1.5.1, NumPy 1.26.4, Python 3.9--3.12.
-   Colab has newer TF/Keras 3 and sklearn preinstalled. Don't `pip install -r requirements.txt` in Colab, which the README suggests: it can downgrade NumPy/TF and break the runtime. Print `tf.__version__`, `keras.__version__` and `sklearn.__version__` in both environments instead.
-   The template's optional cell (`!pip install tensorflow==2.12` with `keras.wrappers.scikit_learn`) is out of date. That wrapper doesn't exist in Keras 3, and TF 2.12 won't install on Colab's current Python. Leave that cell commented out as the template has it.
-   If you choose scikeras anyway, it needs a recent version for Keras 3 and isn't in requirements.txt. Install it in both environments.
-   The template's `from keras.models import Sequential` works under Keras 3 in both places.

GPU and runtime

-   TensorFlow hasn't supported native GPU on Windows since 2.11, so local runs on CPU.
-   The workload is tiny: about 270 rows per training fold, about 9 batches per epoch. 4 models × 3 folds × 100 epochs is about 1,200 epochs, plus Part 1, which takes minutes on CPU.
-   A Colab GPU gives little or no speedup for models this small and can even be slower because of overhead. Use `verbose=0` to keep the output readable.

Reproducibility

-   Set one `SEED` constant, call `keras.utils.set_random_seed(SEED)` before each model, and use the same `random_state` in `KFold`.
-   Optionally call `tf.config.experimental.enable_op_determinism()`.
-   Mention in markdown that results will differ slightly between Windows CPU and Colab, and between TF versions, due to floating-point and kernel differences. The template itself notes this ("Your results may vary...").

Final check before submitting

-   [ ]  Kernel → Restart and Run All passes in VS Code on Windows.
-   [ ]  Runtime → Restart and run all passes in Colab.
-   [ ]  Every output cell is filled in, with all plots and tables visible.
-   [ ]  Your name and student number are filled in (they already are).
-   [ ]  Save and submit the fully executed `.ipynb`.

### 7\. Where your cells go (template cells stay as they are, in the same order)

-   Optional pip cell: leave it commented out.
-   Imports cell: leave it. Add a new code cell right after it for extra imports (numpy, pandas, matplotlib, sys/os, `train_test_split`) and a version printout.
-   Empty code cell under "Assignment Start": environment detection, `SEED`, and constants such as `EPOCHS=100`, `N_FOLDS=3`, `BATCH_SIZE`.
-   New markdown and code cells right after it: "Helper Functions" section with all the functions from section 3.
-   "Data Import" cell: edit minimally (conditional `drive.mount`, detected data folder, real `X_*.csv` names, `index_col=0`) with a short comment explaining the change.
-   "Hint: standardize" cell and the two empty cells after it: shape checks and a note on the scaling decision. No upfront fit, per decision 1.
-   1a, 1b, 1c, 2a--2d: put code in or after the "# your code here" cell, then add a new markdown cell after it for your answer. Keep the template's comment-only cells.
-   2e: comparison table and plot in a code cell after the template comment cell, then a markdown answer.
-   Final clarity note cells: leave them as they are.

### 8\. Resolved questions

| # | Question | Decision |
| --- | --- | --- |
| Q1 | Edit the data-import cell? | Yes. `drive.mount` only in Colab, detected data folder, short comment explaining the change. |
| Q2 | How to evaluate Part 1 | Single shuffled 80/20 `train_test_split` (not `validation_split`), 100 epochs. No CV in Part 1. |
| Q3 | Part 1 hidden activation | ReLU, stated in one line. |
| Q4 | Test set | Optional: test MAE for the best model only, after 2e, retrained on the full training set. |
| Q5 | 1b sizes | 16/8 as written; note the depth + width confound in the write-up. |
| Q6 | Which MAE to report | Final-epoch validation MAE, mean ± std across the 3 folds. |
| Q7 | CV curves | Mean ± std band across folds. |
| Q8 | RMSprop learning rate | Defaults (both 0.001), so only the optimizer changes. |
| Q9 | README mismatches | Later, if at all; not graded. |

Next step: Day 1, step 2 (setup cell and data loading). Once shapes show (404, 13) and (102, 13) in both environments, move on to 1a.