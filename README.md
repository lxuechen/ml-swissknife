# swissknife

Reusable ML research primitives for fast prototyping. 

Slowly porting and refactoring lots of stale code from many past projects.

## Installation

```bash
pip install git+https://github.com/lxuechen/swissknife.git
```

## TODOs
- [x] Numerical utils
  - [x] simultaneous iteration
  - [x] Lanczos
  - [x] spectral density plotting
- [ ] NLP
  - [ ] Decoding utils, new token embedding for GPT-2, fast dataset loading (w/ auto-download)
  - [ ] Evaluation pipeline for generation
- [ ] Custom optimizers (HF, K-FAC)
- [ ] Fast two-sample test utils
- [ ] Helper code for calibration metrics (reliability diagram, ECE, Brier score)
- [x] Boilerplate for fine-tuning GPT-3 via OpenAI API.
- [x] Confidence interval utils
