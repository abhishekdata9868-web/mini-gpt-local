# Project Status / Session Handoff

Last updated: 2026-10-29

## Long-term goal

Build MYAI: a useful autonomous coding assistant that can understand tasks, inspect project files, make bounded changes, run targeted tests, recover from errors, and record decisions/results. Longer-term capabilities include Hindi/Hinglish support, persistent memory, multimodal inputs, app generation, and controlled deployment.

## Current repository

- Repository: https://github.com/abhishekdata9868-web/mini-gpt-local
- Working branch: `fix/mps-training-colab-setup`
- Review: [PR #1](https://github.com/abhishekdata9868-web/mini-gpt-local/pull/1)
- Main branch has not been modified or merged by this work.

## What is confirmed complete

- Added CUDA -> Apple Silicon MPS -> CPU device selection to the from-scratch training script.
- Added `python from_scratch/train.py --smoke-test` for device/forward/backward/optimizer/checkpoint save-load/next-token checks without downloading the training corpus.
- Updated the Colab notebook to clone the repository if needed, install dependencies, run the smoke test, then start training.
- Added NumPy to `from_scratch/requirements.txt`.
- Updated README with separate instructions for the Qwen-based chat app and the from-scratch educational model.
- User ran the smoke test on an Apple Silicon Mac and reported:
  `SMOKE TEST PASSED | device=mps`
  The expected logits shape was `(2, 8, 32)`; next-token batch shape was `(2,)`.

## Known issues / limits

- The full 3,000-step training run has not yet been confirmed.
- The initial smoke-test run warned that NumPy was missing. NumPy has since been added to the branch requirements, but the user's local environment must pull the latest branch and install requirements.
- The user's shell startup displayed `~/.zshrc:5: unmatched "`. This is separate from MiniGPT and should be inspected/fixed when convenient; it did not stop the Python smoke test.
- Smoke test PASS verifies basic execution, not language quality or a successful full training run.

## Next actions

1. In the local repository, update from the working branch:
   `git pull --ff-only`
2. Install the current dependencies:
   `python -m pip install -r from_scratch/requirements.txt`
3. Re-run `python from_scratch/train.py --smoke-test` and confirm it still reports `device=mps`.
4. Start full training with `python from_scratch/train.py`; capture device, periodic train/validation loss, completion status, and checkpoint-save message.
5. Inspect generated sample text and checkpoint integrity. Do not claim the model is good based on a decreasing loss alone.
6. Record verified results here and in the PR. Keep changes on the review branch until reviewed; do not merge to `main` without an explicit decision.

## End-of-session update checklist

- [ ] Summarize what changed and why.
- [ ] Record exact test commands and actual results (PASS/FAIL; do not infer success).
- [ ] Record branch and commit/PR link.
- [ ] List blockers and one concrete next action.
- [ ] Commit and push reviewed changes to the working branch.
