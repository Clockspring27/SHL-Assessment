# Spoken Grammar Scoring Engine

Predicts a continuous grammar score (0 to 5) for 45-60 second spoken-English clips, built for the **SHL Hiring Assessment 2026** Kaggle competition (769 training clips, 216 test clips).

The whole solution is a single, cached, top-to-bottom notebook: [`grammar_scoring_submission.ipynb`](grammar_scoring_submission.ipynb).

## Approach

Each clip is described by several complementary views. Small, heavily regularised regressors are fitted on each view and combined with a non-negative blend.

| Stage | What happens |
|---|---|
| 1. Transcription | Whisper (base) transcribes every clip, keeping word timestamps and confidence statistics. Clips with an empty transcript are kept and flagged, so every test clip gets a prediction. |
| 2. Handcrafted features | Sentence structure and parse depth (spaCy), grammar-error counts (LanguageTool, optional), perplexity (distilgpt2), pauses, fillers, repetitions, Whisper confidence. |
| 3. Embeddings | MiniLM on the transcript; Whisper-small, WavLM-base-plus and WavLM-large hidden states, every layer, mean and std pooled. The best layers are chosen by a per-layer probe. |
| 4. Speaker clusters | Speaker x-vectors are clustered so one voice never sits on both sides of a cross-validation split. |
| 5. Base models | Ridge, SVR and LightGBM on several feature views; regularisation chosen by speaker-grouped CV. |
| 6. Blend and post-processing | Non-negative least squares on out-of-fold predictions, optional speaker smoothing (applied only if it helps in CV), and a no-speech rule for clips with no detectable speech. |

### Why speaker-grouped cross-validation

The same speakers recur across clips, and one speaker's clips receive almost the same score. With random folds the validation set contains speakers the model has already seen, which makes the score look better than it will be on new speakers. All model selection here uses folds that keep each speaker cluster together.

## Results

| Quantity | Value |
|---|---|
| Mean-predictor RMSE (reference) | _fill in from the notebook's first data cell_ |
| Training RMSE (in-sample, final pipeline) | _fill in from the notebook's last results cell_ |
| Speaker-grouped out-of-fold RMSE | _fill in from the notebook's last results cell_ |
| Kaggle public leaderboard score (lower is better) | _fill in after submitting `submission.csv`_ |

For reference, earlier iterations of this project scored on the public leaderboard as follows:

| Iteration | Leaderboard score |
|---|---|
| Whisper transcripts, handcrafted features, MiniLM and a single Whisper-base embedding; Ridge, SVR and LightGBM ensemble | 0.4314 |
| Multi-layer Whisper-small pooling plus WavLM-base-plus, weighted ensemble | 0.4175 |
| Plus speaker-cluster smoothing | 0.4145 |
| Plus WavLM-large and a restructured pipeline | 0.4122 |

The training RMSE is in-sample and is reported because the competition requires it. The speaker-grouped out-of-fold RMSE is the honest estimate of generalisation.

## What did not work

- **Dropping the label-0 clips from training.** A variant trained without them scored worse on the leaderboard (0.4510), so the notebook keeps them by default (`TRAIN_ON_ZEROS = True`).
- **A learned zero-clip detector.** It flagged no test clips. The notebook now uses a CTC-based speech detector and applies it only if it separates the training zeros cleanly.
- **End-to-end fine-tuning of WavLM.** One fold reached about 0.60 validation RMSE at roughly 5 minutes per epoch on a T4, which is no better than the frozen-feature models, so it is not part of the final pipeline.
- **An audio language model (Voxtral-Mini-3B) as an extra view.** Its processor failed on the latest `transformers` release, so it was left out.

## Running it

**Environment:** Google Colab with a T4 GPU, or any machine with a CUDA GPU. Python 3.10+.

1. Download the competition data and set `DATA_DIR` in the config cell (the folder that contains `train/`, `test/`, `train.csv` and `test.csv`). The data is not included in this repository.
2. Install the dependencies (the commented `pip` lines in the first cell):
   `openai-whisper`, `librosa`, `soundfile`, `sentence-transformers`, `transformers`, `torch`, `lightgbm`, `scikit-learn`, `scipy`, `spacy` (with `en_core_web_sm`), and optionally `language_tool_python` (needs Java).
3. Run all cells top to bottom. The notebook writes `submission.csv` (columns `filename`, `label`).

Every expensive step is cached under `cache/`, so re-running is fast. To skip transcription and text features, place an existing `all_df.csv` and `hc.csv` in `cache/` and the notebook loads them (after checking that the rows line up with `train.csv` and `test.csv`).

**Rough cost on a T4:** transcription, text features and the embedding extractors are the slow parts; everything after them (probes, models, blend) runs on CPU in minutes.

## Notebook layout

1. Data
2. Stage 1: transcripts and handcrafted features
3. Stage 2: embeddings
4. Speaker clusters and speaker-grouped folds
5. Modelling helpers and feature blocks
6. Base models
7. Blend, speaker smoothing and the no-speech rule
8. Submission
9. Results and report

## Limitations

- The blend weights, the smoothing strength and the speaker-cluster threshold are chosen on the same out-of-fold predictions or training labels that score them, so the grouped-CV figure is slightly optimistic.
- Speaker clusters come from voice similarity, not ground-truth speaker IDs.
- Whisper tends to normalise grammar in its transcripts, which weakens text-only signals. The audio views compensate.
- The cross-validation estimate and the leaderboard score differ, because the public test set has its own mix of speakers and recording lengths.
