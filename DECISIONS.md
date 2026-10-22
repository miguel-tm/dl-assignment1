# Design and Coding Decisions: SCS 3546 Assignment 1

A running log of the design, modelling and environment decisions for this assignment.
It has two purposes:

1. **Learning:** each entry explains *why* the choice was made and what the alternative was.
2. **Report:** the "Notebook note" line gives a short justification to reuse in a code comment or markdown cell.

**How to maintain this file**
- Add new decisions at the end of their section with the next ID (e.g. D17, E09), using the same format (Decision / Why / Alternative considered / Notebook note), and add them to the Quick index.
- New entries start as `Proposed`. Only mark them `Accepted` once confirmed.
- If a decision changes, don't delete it. Set its status to `Superseded by Dxx`, add the new entry, and log it in the Change log.
- Every change gets a one-line row in the Change log with the date.
- Status values: `Accepted`, `Proposed`, `Superseded`.

---

## Quick index

| ID | Decision | Status |
|---|---|---|
| D01 | Manual KFold loop instead of scikeras `cross_val_score` | Accepted |
| D02 | Fit the scaler on training rows only (deliberate deviation from the template hint) | Accepted |
| D03 | Leave the target `y` unscaled | Accepted |
| D04 | Part 1 validation: shuffled 80/20 `train_test_split`, not `validation_split` | Accepted |
| D05 | Part 1 trains for 100 epochs | Accepted |
| D06 | Part 1 hidden activation is ReLU | Accepted |
| D07 | 1b uses 16/8 as written; the depth + width confound is noted | Accepted |
| D08 | Output layer is `Dense(1)` with linear activation | Accepted |
| D09 | Loss is MSE; MAE is tracked as a metric | Accepted |
| D10 | Batch size and learning rate are left at defaults for every model | Accepted |
| D11 | Same seed and same folds for every model | Accepted |
| D12 | Fresh model per fold: `clear_session` → seed → build | Accepted |
| D13 | Reported CV MAE is the final-epoch validation MAE, mean ± std | Accepted |
| D14 | CV learning curves are plotted as mean ± std bands | Accepted |
| D15 | Test set is used once, for the best model only, after 2e (optional) | Accepted |
| D16 | Input defined with `keras.Input(shape=(13,))` | Accepted |
| D17 | Learning curves use a log-scale y-axis | Superseded by D19 |
| D18 | RMSE reported alongside MSE and MAE in every results table | Accepted |
| D19 | Log y-axis is the default but optional; 1a also shows the linear version | Accepted |
| E01 | Notebook-first project; no `src/` package | Accepted |
| E02 | One setup cell detects Colab vs. local and sets the data path | Superseded by E09 |
| E03 | Use the real filenames (`X_train.csv`, capital X) | Accepted |
| E04 | Read CSVs with `index_col=0` | Accepted |
| E05 | Don't install `requirements.txt` in Colab; print versions instead | Accepted |
| E06 | Local runs on CPU; no GPU needed | Accepted |
| E07 | Small result differences between environments are expected and stated | Accepted |
| E08 | Numbers quoted in markdown match the outputs of the submitted run | Accepted |
| E09 | Data folder = first candidate containing `X_train.csv`; otherwise a clear error listing the paths tried | Accepted |
| E10 | Library versions are a guide; Keras must be 3.x; actual versions recorded | Accepted |
| E11 | CSVs are read in the template's data-import cell; no `load_data` helper | Accepted |
| E12 | Submitted outputs come from local VS Code; Colab is a portability check only | Accepted |
| P01 | Copilot scaffolds and advises; the solution is my own work | Superseded by P04 |
| P02 | Required parts first; optional extras only if time allows | Accepted |
| P03 | Each helper is written right before its first use | Accepted |
| P04 | Copilot explains each step and writes the code; I review, test and commit; analysis is mine | Accepted |
| P05 | Rough notes right after each model run; Phase 4 polishes them | Accepted |
| P06 | No decision IDs in notebook comments; plain-language reasons instead | Accepted |

---

## Modelling and experiment design

### D01: Manual KFold loop instead of scikeras `cross_val_score`
- **Decision:** Use `KFold(n_splits=3, shuffle=True, random_state=SEED)` in a manual loop and keep each fold's `history`.
- **Why:** Part 2 needs both CV scores *and* per-epoch learning curves. `cross_val_score` only returns final scores, not histories. The template's `keras.wrappers.scikit_learn` no longer exists in Keras 3.
- **Alternative considered:** scikeras `KerasRegressor` + `cross_val_score`. It's simpler for scores, but you'd need a second pass to get the curves, plus an extra dependency.
- **Notebook note:** "A manual KFold loop is used so each fold's training history is available for the learning curves; `cross_val_score` returns only final scores."

### D02: Fit the scaler on training rows only (deliberate deviation from the template hint)
- **Decision:** Part 1: fit `StandardScaler` on the 80% training split only. Part 2: fit it inside each fold, on that fold's training rows only.
- **Why:** If the scaler is fitted on all the data before splitting, the validation rows' mean and std leak into training (*data leakage*). The validation score then looks slightly better than it really is.
- **Alternative considered:** Standardize upfront, as the template hint suggests. It's simpler, and the effect on this small dataset is probably small, but it's methodologically weaker.
- **Notebook note:** "The template suggests standardizing upfront; I fit the scaler on training rows only (per fold in CV) to avoid leaking validation statistics into training."

### D03: Leave the target `y` unscaled
- **Decision:** Scale only the features. The target stays in thousands of dollars.
- **Why:** MAE and RMSE can then be read directly as dollar errors (MAE = 3.0 means about $3,000). With 13 features and a target of moderate size, training stays stable.
- **Alternative considered:** Scale `y` too. It can smooth optimization, but every metric would have to be converted back before you could interpret it.
- **Notebook note:** "The target is left unscaled so MAE/RMSE are directly interpretable in $1000s."

### D04: Part 1 validation uses a shuffled 80/20 `train_test_split`, not `validation_split`
- **Decision:** `train_test_split(X, y, test_size=0.2, random_state=SEED)`, then `fit(..., validation_data=(X_val, y_val))`.
- **Why:** Keras `validation_split` takes the *last* 20% of rows without shuffling, so any ordering in the file biases validation. It also makes it hard to fit the scaler on training rows only (D02).
- **Alternative considered:** `validation_split=0.2`. It's one argument shorter, but it isn't shuffled and doesn't fit with D02.
- **Notebook note:** "A shuffled hold-out split is used instead of `validation_split`, which takes the last rows unshuffled."

### D05: Part 1 trains for 100 epochs
- **Decision:** 100 epochs for 1a–1c.
- **Why:** The template doesn't specify epochs for Part 1. Matching Part 2 keeps the experiments consistent and comparable.
- **Notebook note:** "100 epochs, matching Part 2, for consistency."

### D06: Part 1 hidden activation is ReLU
- **Decision:** ReLU in all Part 1 hidden layers.
- **Why:** The template doesn't name one. ReLU is the standard default, avoids vanishing gradients, and matches Models 1–2 in Part 2.
- **Notebook note:** "The template does not specify an activation for Part 1; ReLU is used as the standard default."

### D07: 1b uses 16/8 as written; the depth + width confound is noted
- **Decision:** Use the template's example sizes exactly: 1b = 16→8, 1c = 32→16.
- **Why:** Sticking to the spec avoids losing marks. 1a (13) → 1b (16→8) changes **depth and width**, so that comparison is confounded. 1b → 1c changes **width only**, so it's a clean comparison. Saying this explicitly shows experimental awareness.
- **Notebook note:** "Note: 1a→1b changes both depth and width; 1b→1c isolates width."

### D08: Output layer is `Dense(1)` with linear activation
- **Decision:** One output unit with no activation (linear).
- **Why:** This is regression with an unbounded continuous target. An activation such as sigmoid or ReLU would wrongly limit the output range.
- **Notebook note:** "Linear single-unit output for an unbounded regression target."

### D09: Loss is MSE; MAE is tracked as a metric
- **Decision:** `loss="mse"`, `metrics=["mae"]`.
- **Why:** The template specifies MSE. MSE penalizes large errors more heavily and is smooth to optimize. MAE is easier to interpret (average error in $1000s), and it's what Part 2 asks you to report.

### D10: Batch size and learning rate left at defaults for every model
- **Decision:** Batch size 32 (Keras default). Learning rate 0.001, the default for both Adam and RMSprop.
- **Why:** Neither is in the spec. Keeping them identical everywhere means each comparison changes only the intended variable. Model 4 vs. Model 2 then changes **only the optimizer**.
- **Notebook note:** "Batch size and learning rate are kept at defaults for all models so only the intended hyperparameter changes."

### D11: Same seed and same folds for every model
- **Decision:** One `SEED` constant, one `KFold` with a fixed `random_state`, and the seed is reset before each model.
- **Why:** Differences between models should come from the architecture or hyperparameter, not from luck in the split or the weight initialization (a *controlled comparison*). With the same seed, models with identical layer shapes start from the same initial weights in each fold and see batches in the same order. Models 2, 3 and 4 are all 128/64, so Model 2 vs. 3 differs only in the activation and Model 2 vs. 4 only in the optimizer. Model 1 (64/64) has different layer shapes, so its initial weights differ anyway.
- **Limitation:** The ± std reflects variability across *data splits* (folds) only, not across *random weight initializations*. State this in 2e.
- **Alternative considered:** A different seed per model, or several seeds per model. That would measure initialization variance too, but it multiplies the runs and blurs the one-variable comparisons.
- **Notebook note:** "All models share a fixed seed, so variability across weight initializations is not measured; the ± std reflects variation across CV folds only."

### D12: Fresh model per fold: `clear_session` → seed → build
- **Decision:** In each fold, call `keras.backend.clear_session()`, then `keras.utils.set_random_seed(SEED)`, then build the model.
- **Why:** A new model per fold means no weights carry over between folds. Clearing *before* seeding makes sure the seed applies to the new graph.

### D13: Reported CV MAE is the final-epoch validation MAE, mean ± std
- **Decision:** Take the validation MAE at epoch 100 for each fold, then report the mean ± std across the 3 folds.
- **Why:** Picking the "best epoch" uses the validation data to choose the result, which is optimistic and a quiet form of leakage. The std shows how stable the model is across splits.
- **Notebook note:** "CV MAE = final-epoch validation MAE averaged over 3 folds (± std); best-epoch selection is avoided because it would tune on validation data."

### D14: CV learning curves plotted as mean ± std bands
- **Decision:** One line per metric (training/validation) for the mean across folds, with a shaded ±std band.
- **Why:** This is cleaner than 3 overlapping lines per metric, and it shows the same variability as the reported ± std.

### D15: Test set used once, for the best model only, after 2e (optional)
- **Decision:** After choosing the best model with CV, retrain it on the full training set, then report test MAE once.
- **Why:** The test set must not influence model selection, or it stops being an unbiased estimate. It's optional because the template doesn't require it.

### D16: Input defined with `keras.Input(shape=(13,))`
- **Decision:** Use an explicit `Input` layer instead of `input_dim` / `input_shape` on `Dense`.
- **Why:** Keras 3 warns about the old style. The explicit input works the same locally and in Colab.

### D17: Learning curves use a log-scale y-axis
- **Decision:** Both panels (MSE and MAE) of every learning-curve plot use a log y-axis.
- **Why:** MSE starts near 600 and ends around 10–40. On a linear axis the last 50+ epochs look flat, which hides the train/validation gap that the questions are about.
- **Alternative considered:** Linear axis, optionally starting from epoch 10. More familiar to read, but the first epochs still dominate the range.
- **Notebook note:** "Learning curves use a log scale so differences in later epochs stay visible."

### D18: RMSE reported alongside MSE and MAE in every results table
- **Decision:** Part 1 and Part 2 tables show training and validation MSE, RMSE and MAE.
- **Why:** MSE is in squared $1000s and hard to read. RMSE = √MSE is back in $1000s, so it compares directly with MAE. The template's preamble suggests this. The gap RMSE − MAE also hints at how much large errors contribute.
- **Alternative considered:** MSE and MAE only. Fewer columns, but the loss has no intuitive unit.
- **Notebook note:** "RMSE = √MSE is shown so the loss can be read in $1000s, like MAE."

### D19: Log y-axis is the default but optional; 1a also shows the linear version
- **Decision:** `plot_learning_curves` has a `log_scale_y_axis` flag (default `True`). 1a shows both the linear and the log plot; the other models show the log plot only.
- **Why:** The linear plot shows the familiar shape of fast early learning; the log plot keeps the late-epoch gap visible. Showing both once explains why log is used afterwards.
- **Alternative considered:** D17 as written (log only everywhere).
- **Notebook note:** "1a shows both scales; later plots use the log scale only."

---

## Environment and tooling

### E01: Notebook-first project; no `src/` package
- **Decision:** All helper functions live in a "Helper Functions" section inside the notebook.
- **Why:** The deliverable is one self-contained, fully executed `.ipynb`. A `src/` package would break in Colab and for graders unless it's uploaded too, and the two copies could drift apart.

### E02: One setup cell detects Colab vs. local and sets the data path
- **Decision:** Detect Colab with `"google.colab" in sys.modules`. In Colab, mount Drive (or use `/content/`). Locally, use whichever of `Path("data")` or `Path("../data")` exists.
- **Why:** The template's unconditional `drive.mount` crashes locally. The fallback covers VS Code kernels that start from either the repo root or `notebooks/`.
- **Notebook note:** "Data import edited so it runs both in Colab and locally (the template invites updating paths)."

### E03: Use the real filenames (`X_train.csv`, capital X)
- **Why:** The template reads `x_train.csv`. Windows ignores case, but Colab (Linux) doesn't, so the template's name fails there.

### E04: Read CSVs with `index_col=0`
- **Why:** Each CSV starts with an unnamed index column. Without this you'd get 14 "features" instead of 13. Check shapes: (404, 13) train, (102, 13) test.

### E05: Don't install `requirements.txt` in Colab; print versions instead
- **Decision:** Use Colab's preinstalled packages. Print `tf`, `keras` and `sklearn` versions in both environments. Leave the template's `tensorflow==2.12` cell commented out.
- **Why:** Installing pinned versions in Colab can downgrade NumPy/TF and break the runtime. TF 2.12 doesn't install on Colab's current Python.

### E06: Local runs on CPU; no GPU needed
- **Why:** TensorFlow has had no native Windows GPU support since 2.11. The workload is tiny (~1,200 epochs on ~270 rows per fold), so a GPU wouldn't help and can be slower because of overhead. Use `verbose=0` to keep output readable.

### E07: Small result differences between environments are expected and stated
- **Why:** Floating-point and kernel differences between Windows CPU and Colab, and between TF versions, change the numbers slightly even with fixed seeds. The template itself says results may vary.

### E08: Numbers quoted in markdown match the outputs of the submitted run
- **Decision:** Choose one environment for the final "Restart and Run All", and write or refresh the markdown numbers from that run.
- **Why:** The grader sees the outputs and the text side by side, so they must agree.

### E09: Data folder = first candidate containing `X_train.csv`; otherwise a clear error
- **Decision:** Try the candidate folders in order (Colab folder, `data`, `../data`) and use the first one that contains `X_train.csv`. If none does, raise an error that lists every path tried.
- **Why:** E02 only checks that a folder exists, so an empty or wrong `data/` folder would pass and fail later with a confusing pandas error. Listing the tried paths makes the problem obvious at a glance.
- **Alternative considered:** E02 as written (folder existence only). It's slightly shorter but fails late and unclearly.
- **Notebook note:** "The data folder is the first candidate that contains the CSVs; otherwise an error lists the paths tried."

### E10: Library versions are a guide; Keras must be 3.x
- **Decision:** Accept whatever versions are installed (locally from `requirements.txt`, in Colab preinstalled) as long as Keras is 3.x. Record the printed versions for both environments below.
- **Why:** The code relies on Keras 3 behaviour (D16, D01), but nothing needs an exact TF or sklearn version. Chasing exact matches costs time and conflicts with E05.
- **Alternative considered:** Pin identical versions in both environments. It would reduce E07 differences slightly, but risks breaking Colab's runtime.
- **Recorded versions:** Local (Windows, VS Code): Python 3.12.7, TensorFlow 2.16.2, Keras 3.15.1, scikit-learn 1.5.1, NumPy 1.26.4, pandas 2.2.2. Colab: Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2, scikit-learn 1.6.1, NumPy 2.1.3, pandas 2.2.3 (checked 2026-10-22).
- **Notebook note:** "Library versions are printed at the top; small result differences between environments are expected."

### E11: CSVs are read in the template's data-import cell; no `load_data` helper
- **Decision:** Edit the template's data-import cell in place: mount Drive only in Colab, find the folder with `get_data_dir()`, and read the four CSVs into the template's `df_*` names. The `load_data` helper listed in P03 is not written.
- **Why:** The data is read exactly once, so a helper adds a layer without reuse. Keeping the reads in the template cell shows the grader the minimal change to their code.
- **Alternative considered:** A `load_data(data_dir)` helper returning four arrays. Slightly tidier, but a one-time abstraction that hides the template's cell.
- **Notebook note:** "Data import edited so it runs both in Colab and locally."

### E12: Submitted outputs come from local VS Code; Colab is a portability check only
- **Decision:** The final "Restart and Run All" (E08) is done locally in VS Code, and that executed notebook is submitted. Colab is used only to confirm the notebook still runs there, in case the instructor re-runs it.
- **Why:** Outputs are saved straight into the committed notebook, versions are pinned and recorded, and there are no session timeouts or Drive download steps. The data import already works in Colab (checked in 1.4).
- **Alternative considered:** Final run in Colab, the instructor's reference environment. It needs a download back into the repo, and Colab's versions change without notice.
- **Notebook note:** "Outputs were produced locally (versions printed above); the notebook also runs in Colab."

---

## Process

### P01: Copilot scaffolds and advises; the solution is my own work
- **Why:** Assignments are individual work. Copilot is used for repo setup, planning and debugging help, not for writing the solution or the analysis.

### P02: Required parts first; optional extras only if time allows
- **Why:** The submission is already late. The 2e table and the markdown write-ups earn more marks than extra plots or the optional test evaluation.

### P03: Each helper is written right before its first use
- **Decision:** Step 2 writes only `set_global_seed`, `get_data_dir` and `load_data`. The Part 1 helpers come in step 3 and the CV helpers in step 6.
- **Why:** It keeps step 2 small (about 1h), reaches a working data load sooner, and each helper is tested as soon as it's written.
- **Alternative considered:** Write the whole Helper Functions section up front. That means more untested code before anything runs.

### P04: Copilot explains each step and writes the code; I review, test and commit; analysis is mine
- **Decision:** One sub-phase at a time. Copilot explains the concepts and structure, then writes the code. I review it, test it, commit, and say "next". The analysis and conclusions in markdown are written by me.
- **Why:** I'm orchestrating the work, and that's OK with the instructor. It's faster under the deadline while I still understand and own every step.
- **Alternative considered:** P01 (Copilot advises only; I write all the code). Slower given the late submission.

### P05: Rough notes right after each model run; Phase 4 polishes them
- **Decision:** After each model runs (1a–1c, 2a–2d, 2e), write 2–3 rough bullets in its markdown placeholder about what the curves and numbers show. Phase 4 turns them into full answers.
- **Why:** Observations are freshest right after the run, I don't have to recall 7 sets of curves at the end, and every question has something under it if time runs out.
- **Alternative considered:** Write all the analysis in Phase 4. Harder to recall and riskier on time.

### P06: No decision IDs in notebook comments; plain-language reasons instead
- **Decision:** New notebook code states reasons in plain words. Existing IDs such as "(D05)" or "(E10)" are replaced during the clarity pass (6.1).
- **Why:** The grader never sees DECISIONS.md, so the IDs are meaningless to them and cost clarity marks. The log stays the place for the full reasoning.
- **Alternative considered:** Keep the IDs and add a key in the notebook. More text for the grader with no benefit.

---

## Change log

| Date | Change |
|---|---|
| 2026-10-21 | Initial log created from PLAN.md and the planning discussion (D01–D16, E01–E08, P01–P02). |
| 2026-10-22 | D11: added identical-starting-weights rationale, ± std limitation, alternative and notebook note. Added E09 and P03 as Proposed. Maintenance rules expanded. |
| 2026-10-22 | E09 and P03 accepted; E02 superseded by E09. |
| 2026-10-22 | P01 superseded by P04 (Copilot writes code, I review). Added P05 (rough notes after each run) and E10 (versions are a guide), all Accepted. |
| 2026-10-22 | E10: recorded local versions (TF 2.16.2, Keras 3.15.1). |
| 2026-10-22 | Added P06 (no decision IDs in notebook comments), Accepted. Added E11 (no `load_data` helper) as Proposed. |
| 2026-10-22 | E11 accepted. |
| 2026-10-22 | E10: recorded Colab versions (TF 2.21.0, Keras 3.13.2). Added E12 (local VS Code produces the submitted outputs), Accepted. |
| 2026-10-22 | Added D17 (log-scale learning curves) as Proposed. |
| 2026-10-22 | D17 accepted. |
| 2026-10-22 | Added D18 (RMSE in results tables), Accepted at my request. Added D19 (optional log scale, both scales in 1a) as Proposed; D17 is superseded if D19 is accepted. |
| 2026-10-22 | D19 accepted; D17 superseded by D19. |
