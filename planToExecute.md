## Implementation Plan: Notebook in Small Sub-phases

This plan covers 20 sub-phases in 6 phases, each 15–45 minutes. Every sub-phase ends with a notebook that passes "Restart and Run All" up to that point (empty template cells are fine).

Cell numbers refer to the original template, which has 36 cells. They will shift once you insert new cells, so each step also names the section.

**Markers**
- 🟦 **Colab check needed**, not only VS Code.
- 🟨 **Decision flag**: update DECISIONS.md in the same commit.

**Execution rule:** one sub-phase at a time. After each one, I stop until you've tested it, committed it, and said "next".

---

### Phase 1: Setup and data → tag `v0.1-data-loaded`

**1.1 Imports, versions, constants** (15–20 min)
- **Goal:** Add the extra imports, print library versions, and define all constants in one place.
- **Cells touched:**
  - Cell 6 (template imports): keep it as is.
  - New code cell right after it: numpy, pandas, matplotlib, sys, `pathlib`, `train_test_split`, and a version printout.
  - Cell 10 (the empty cell under "Assignment Start"): `SEED`, `EPOCHS=100`, `N_FOLDS=3`, `BATCH_SIZE=32`, `VAL_SIZE=0.2`.
- **What to understand:**
  - Named constants are what make "same settings for every model" (D10, D11) checkable at a glance.
  - Printing versions explains environment differences later (E05, E07).
- **How to test:** Restart and Run All. Versions print (TF 2.16.x / Keras 3.x locally) and there are no warnings about missing modules.
- **Commit:** `chore(setup): add imports, version printout and global constants`

**1.2 Helper section: seed and data folder** (20–30 min)
- **Goal:** Start the "Helper Functions" section with `set_global_seed` and `get_data_dir`.
- **Cells touched:** new markdown heading and new code cell right after cell 10.
- **What to understand:**
  - `keras.utils.set_random_seed` seeds Python, NumPy and TF in one call (D11).
  - The candidate folders are tried in order (Colab folder, data, `../data`). The first one that contains X_train.csv wins. If none does, the error lists every path tried (E09).
  - The Colab folder (a Drive path or `/content`) is a constant you set once.
- **How to test:**
  - `get_data_dir()` prints the absolute path of data.
  - Temporarily call it with a bogus candidate list and confirm the error names every path tried. Then remove that test.
- **Commit:** `feat(helpers): add seed and data-folder detection helpers`

**1.3 Data import and shape checks** (20–30 min) 🟨
- **Goal:** Make the template's import cell run in both environments and produce NumPy arrays.
- **Cells touched:**
  - Cell 12 (Data Import):
    - Run `drive.mount` only when running in Colab.
    - Read from the detected folder, using the real `X_*.csv` names and `index_col=0`.
    - Add a one-line comment explaining why the cell was edited (E02/E09).
    - Keep the template's `df_*` variable names.
  - Cells 13–15 (hint plus two empty cells): convert to arrays (`y` as 1-D) and print shapes. No scaling here (D02).
- **What to understand:**
  - Why there's an index column, and why you'd otherwise get 14 features (E04).
  - Why Linux is case-sensitive about filenames (E03).
  - Why `y` should be 1-D (`ravel`) so Keras and the metrics don't broadcast unexpectedly.
- **How to test:**
  - Shapes: `(404, 13)`, `(102, 13)`, `(404,)`, `(102,)`.
  - `y_train` ranges from about 5 to 50, with a mean around 22–23.
  - Column count is 13, not 14.
- **🟨 Decision:** PLAN.md lists a `load_data` helper, but the template's own cell already reads the CSVs. My recommendation is to drop `load_data` and keep the reading in the template cell. That stays closest to the template and avoids a helper that's used only once. Record this as a new Proposed entry, or keep `load_data` if you prefer.
- **Commit:** `feat(data): load CSVs from detected folder and verify shapes`

**1.4 Colab check** (15–30 min) 🟦 🟨
- **Goal:** Confirm everything so far also runs in Colab.
- **Cells touched:** none, unless a fix is needed.
- **What to understand:**
  - In Colab, `drive.mount` asks for authorization during Run All.
  - The data has to be uploaded to `/content` or placed in the Drive folder you set as a constant.
  - Expect version differences between environments (E05).
- **How to test:** In Colab, Restart and run all, then check the same shapes as in 1.3 and note the TF, Keras and sklearn versions.
- **🟨 Decision:** pick the environment for the final submitted run now (an extension of E08). Markdown numbers must come from that run, and choosing early avoids rewriting them later.
- **Commit:** `chore(env): verify setup and data loading in Colab` → **tag `v0.1-data-loaded`**

---

### Phase 2: Part 1 → tag `v0.2-part1-done`

**2.1 Part 1 helpers** (30–45 min)
- **Goal:** Add `build_model`, `scale_features`, `train_with_validation` and the single-run mode of `plot_learning_curves`.
- **Cells touched:** add them to the Helper Functions section.
- **What to understand:**
  - Use `keras.Input(shape=(13,))`, a linear `Dense(1)` output, MSE loss and an MAE metric (D08, D09, D16).
  - The scaler is fitted on the 80% training split only, then applied to both splits (D02, D04).
  - Before each model: `clear_session`, then seed, then build (D12).
- **How to test:**
  - `build_model([13]).summary()` shows **196** parameters. 16/8 → **369**, 32/16 → **993**.
  - After `scale_features`, each training column has mean ≈ 0 and std ≈ 1. The validation columns are close to that but not exact. That's expected and shows the scaler wasn't fitted on them.
- **Commit:** `feat(helpers): add model builder, scaling, single-split training and curve plotting`

**2.2 1a: Baseline** (20–30 min)
- **Goal:** Train the 13-unit ReLU model for 100 epochs, then plot the loss curves and report training/validation MSE (and RMSE).
- **Cells touched:**
  - Cell 18: code.
  - Cell 19 (template comment cell): keep it.
  - New empty markdown cell after it, a placeholder for your write-up.
- **What to understand:**
  - Adam on an MSE loss.
  - How to read learning curves: both curves falling means it's still learning; validation flattening while training keeps falling means overfitting.
  - Why there's no output activation (D05, D06, D08).
- **How to test:**
  - Initial loss is around 500–600 (roughly the mean of y² when predictions are near 0).
  - Curves fall steadily.
  - Final validation MAE is roughly 3–5. If it's above about 8, check the scaling and the feature count.
  - Runtime is about 5–15 s on CPU.
- **Commit:** `feat(part1): add baseline 13-unit model and learning curves`

**2.3 1b: Deeper (16→8)** (15–20 min)
- **Goal:** Same pipeline with a 16→8 model.
- **Cells touched:** cell 21, plus a markdown placeholder.
- **What to understand:** going from 1a to 1b changes depth *and* width, so the comparison is confounded (D07).
- **How to test:** 369 parameters, same split as 1a (print the validation size: 81 rows), curves rendered.
- **Commit:** `feat(part1): add deeper 16-8 model`

**2.4 1c: Wider (32→16)** (15–20 min) 🟦
- **Goal:** Same pipeline with a 32→16 model.
- **Cells touched:** cell 23, plus a markdown placeholder.
- **What to understand:**
  - Going from 1b to 1c isolates width (D07).
  - More capacity means a larger gap between training and validation curves is possible.
- **How to test:** 993 parameters. Validation MAE is usually similar to or slightly better than 1b. Run a quick Colab Run All to confirm Part 1 works there.
- **Commit:** `feat(part1): add wider 32-16 model` → **tag `v0.2-part1-done`**

---

### Phase 3: Part 2 → tag `v0.3-part2-done`

**3.1 CV helpers** (30–45 min)
- **Goal:** Add `run_cross_validation`, `summarize_results`, and the mean ± std band mode of `plot_learning_curves`.
- **Cells touched:** Helper Functions section.
- **What to understand:**
  - Inside each fold: fit the scaler, then clear session, seed, build and train (D02, D12).
  - Use one `KFold` with a fixed `random_state` for every model (D01, D11).
  - The reported MAE is the final-epoch validation MAE per fold, then the mean ± std across folds (D13, D14).
- **How to test:**
  - Run a temporary smoke test with `epochs=2`: 3 histories, each 2 epochs long; validation fold sizes are 135/135/134; no index overlap between training and validation.
  - Delete the smoke-test cell before committing.
- **Commit:** `feat(helpers): add k-fold CV loop, result summary and band plots`

**3.2 2a: Model 1 (64/64 ReLU, Adam)** (20–30 min)
- **Goal:** Run 3-fold CV for 100 epochs, plot loss and MAE, and report CV MAE ± std.
- **Cells touched:** cell 26 (keep its comments), plus a markdown placeholder. Store the result in a `cv_results` dict for 2e.
- **What to understand:** what the ± std means: it measures variation across splits, not across random starting weights (D11 limitation).
- **How to test:**
  - 5,121 parameters.
  - CV MAE is roughly 2.3–3.5, with std around 0.1–0.6.
  - Runtime is about 30–60 s on CPU.
- **Commit:** `feat(part2): add Model 1 (64-64 relu) with 3-fold CV`

**3.3 2b: Model 2 (128/64 ReLU)** (15–20 min)
- **Goal:** Same CV pipeline with 128/64.
- **Cells touched:** cell 28, plus a markdown placeholder.
- **How to test:** 10,113 parameters; MAE in a similar range to Model 1.
- **Commit:** `feat(part2): add Model 2 (128-64 relu)`

**3.4 2c: Model 3 (tanh)** (15–20 min)
- **Goal:** Model 2 with tanh activations.
- **Cells touched:** cell 30, plus a markdown placeholder.
- **What to understand:**
  - tanh is zero-centred, bounded, and can saturate.
  - It starts from the same weights as Model 2, so only the activation differs (D11).
- **How to test:** 10,113 parameters. Curves may fall more slowly early on. Results can be similar to Model 2 or somewhat worse.
- **Commit:** `feat(part2): add Model 3 (128-64 tanh)`

**3.5 2d: Model 4 (RMSprop)** (15–20 min)
- **Goal:** Model 2 trained with RMSprop.
- **Cells touched:** cell 32, plus a markdown placeholder.
- **What to understand:**
  - Adam is roughly RMSprop plus momentum plus bias correction.
  - The learning rate is the same (0.001) and the starting weights are the same, so only the optimizer changes (D10, D11).
- **How to test:** 10,113 parameters; curves may be noisier than Adam's.
- **Commit:** `feat(part2): add Model 4 (128-64 relu, rmsprop)`

**3.6 2e: Comparison table and plot** (30–40 min) 🟦
- **Goal:** Add `plot_model_comparison` and a 4-row summary table (CV MAE mean, CV MAE std, CV MSE).
- **Cells touched:**
  - Helper section (the new function).
  - Cell 34: keep the template comment and add code after it.
- **What to understand:** whether differences are smaller than the std. If they are, the models are effectively tied, and your write-up should say so.
- **How to test:** the table numbers match what 2a–2d printed, and the error bars appear. Do a Colab Run All of the whole notebook.
- **Commit:** `feat(part2): add model comparison table and plot` → **tag `v0.3-part2-done`**

---

### Phase 4: Markdown write-ups (you write these; I only review if you ask, per P01)

| Sub-phase | Cells | Key content | Commit |
|---|---|---|---|
| 4.1 Approach notes (20–30 min) | Hint cell area; after the data import | Opens with "Deliberate deviation from the template hint:" (D02); environment-variance note (E07); unscaled target (D03) | `docs(setup): explain preprocessing and environment choices` |
| 4.2 Part 1 (30–45 min) | Placeholders after 1a, 1b, 1c | Curve reading, depth vs width confound (D07), train/val gap | `docs(part1): add analysis for 1a-1c` |
| 4.3 Part 2 models (30–45 min) | Placeholders after 2a–2d | One-variable comparisons (D10, D11), what tanh and RMSprop change | `docs(part2): add analysis for models 1-4` |
| 4.4 2e answer (30–45 min) | After the comparison | Best model and why; the limitation sentence from D11's notebook note | `docs(part2): add model comparison conclusion` |

**How to test:** Restart and Run All, then check that every number quoted in markdown matches the outputs (E08).

---

### Phase 5: Optional test-set evaluation

**5.1 Test MAE for the best model** (20–30 min) 🟨
- **Goal:** Add `evaluate_on_test`: retrain the best model on all 404 rows (scaler fitted on them), then report test MAE once.
- **Cells touched:** helper section, plus a new code and markdown pair after 2e.
- **What to understand:** the test set is used only after the model is chosen (D15).
- **How to test:** test MAE is in the same ballpark as CV MAE (within about ±1). A much lower value is a warning sign.
- **🟨 Decision:** if you skip this, add a Change log note that D15 wasn't exercised.
- **Commit:** `feat(eval): report test MAE for best model`

---

### Phase 6: Final → tag `v1.0-submitted`

**6.1 Clarity pass** (30 min)
- **Goal:** One-line docstrings, consistent names and plot style, no dead code or debug prints.
- **How to test:** Restart and Run All is clean, with no stray warnings you haven't explained.
- **Commit:** `refactor: clarity pass on helpers and plots`

**6.2 Final runs in both environments** (30–45 min) 🟦 🟨
- **Goal:** Restart and Run All in VS Code and in Colab, then save the fully executed notebook from the chosen environment.
- **How to test:** go through the final checklist in PLAN.md section 6; markdown numbers match the outputs.
- **🟨 Decision:** record which environment produced the submitted run (E08).
- **Commit:** `chore(release): final executed notebook` → **tag `v1.0-submitted`**

---

**Total:** about 20 sub-phases, roughly 9–11 hours, consistent with PLAN.md.