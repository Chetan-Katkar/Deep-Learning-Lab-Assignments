# Deep Learning Lab Assignments

Colab-ready notebooks for the Deep Learning practical list (Department of Computer Engineering).

## Experiments 4 – 9

| # | Notebook | Problem statement | Module / Topic | CO |
|---|---|---|---|---|
| 4 | [`Exp04_Traffic_Sign_Recognition_and_Density.ipynb`](Exp04_Traffic_Sign_Recognition_and_Density.ipynb) | Smart city AI system to recognize traffic signs and estimate traffic density from road images on a noisy dataset | 2 · Regularization | CO2 |
| 5 | [`Exp05_CNN_CIFAR10_LeNet_Custom.ipynb`](Exp05_CNN_CIFAR10_LeNet_Custom.ipynb) | Design a CNN (LeNet / custom architecture) to classify CIFAR-10 images | 4 · Neural Networks | CO3 |
| 6 | [`Exp06_Bloom_Taxonomy_Question_Classification.ipynb`](Exp06_Bloom_Taxonomy_Question_Classification.ipynb) | Automatic question paper categorization according to Bloom's Taxonomy | 4 · Neural Networks (NLP) | CO4 |
| 7 | [`Exp07_LSTM_GRU_Stock_Prediction_and_Sentiment.ipynb`](Exp07_LSTM_GRU_Stock_Prediction_and_Sentiment.ipynb) | Train an LSTM/GRU model to predict stock prices or perform sentiment analysis | 3 · Recurrent Networks | CO1 |
| 8 | [`Exp08_Denoising_Autoencoder_MNIST.ipynb`](Exp08_Denoising_Autoencoder_MNIST.ipynb) | Build and train a denoising autoencoder to reconstruct corrupted images (noisy MNIST) | 4 · Autoencoder | CO5 |
| 9 | [`Exp09_Medical_Report_Summarization.ipynb`](Exp09_Medical_Report_Summarization.ipynb) | AI-based medical report summarization for a healthcare organization | 5 · GenAI / Encoder–Decoder | CO6 |

## Running on Google Colab

1. Open the notebook (click the filename above, then **Open in Colab**, or upload it to
   [colab.research.google.com](https://colab.research.google.com)).
2. **Runtime → Change runtime type → Hardware accelerator: `T4 GPU`**.
3. **Runtime → Run all.**

Every notebook is self-contained:

* Only libraries already present on Colab are used (PyTorch, torchvision, scikit-learn, pandas,
  matplotlib, transformers); anything else is `pip install`-ed by the first cell.
* Each notebook prints the detected GPU in its setup cell and warns if the T4 is not enabled.
* Mixed-precision (AMP) training is used where it helps, so the T4's tensor cores are utilised.
* **Every dataset has an offline fallback** — if a download is blocked (GTSRB, yfinance, IMDB,
  HuggingFace hub), the notebook generates an equivalent dataset locally and keeps running, so
  no cell ever fails for want of network access.

Approximate runtime per notebook on a T4: **10 – 25 minutes**. Each notebook exposes an `EPOCHS`
constant near the training cell — raise it for stronger results, lower it for a quicker demo.

## What each notebook contains

Beyond the core model, every notebook answers the five sub-tasks from the practical list in
markdown, and backs the discussion with code — ablations, metric tables, confusion matrices,
attention/feature visualisations and an honest reading of the results.

| Notebook | Highlights |
|---|---|
| **Exp 4** | LeNet→ResNet architecture survey · injected input **and** label noise · six-way regularization ablation · ImageNet transfer learning · macro-F1, balanced accuracy, ECE + reliability diagram · separate density-regression head · TorchScript export with measured batch-1 latency |
| **Exp 5** | LeNet-5 vs custom VGG-style CNN vs compact ResNet at an identical budget · confusion matrix and most-confused pairs · misclassification gallery · first-layer filters, feature maps, t-SNE of the embedding |
| **Exp 6** | Preprocessing pipeline · TF-IDF baseline · BiLSTM with attention pooling vs from-scratch Transformer vs fine-tuned DistilBERT · **QWK and MAE-in-levels** (Bloom levels are ordinal) · adversarial questions that break verb-lookup heuristics |
| **Exp 7** | Stock forecasting **and** sentiment analysis · RNN vs LSTM vs GRU on both · chronological split with the scaler fit on train only · naive-baseline comparison and directional accuracy · measured gradient-flow-by-timestep plot |
| **Exp 8** | Five corruption models · dense vs convolutional DAE · median/Gaussian filter baselines · MSE/PSNR/SSIM · bottleneck sweep · cross-noise generalisation · latent t-SNE and interpolation · **downstream classifier-accuracy recovery** · anomaly detection via reconstruction error |
| **Exp 9** | Self-attention implemented from scratch (+ why √d_k matters, measured) · Seq2Seq GRU with Bahdanau attention and its alignment plot · from-scratch Transformer · zero-shot and fine-tuned T5 · extractive baselines · ROUGE **plus** entity-recall and negation-flip factuality metrics · a working PHI de-identification pass with a corpus-wide leakage audit |

## Notes

* Experiments 6 and 9 generate their corpora inside the notebook (question bank, discharge
  summaries), so they are fully reproducible and contain no real personal or patient data. Each
  has a clearly marked cell to swap in your own CSV.
* Experiment 7 Part A is a deep-learning exercise, **not investment advice**.
* Experiment 9 is **not validated for clinical use**; all data in it is synthetic.
