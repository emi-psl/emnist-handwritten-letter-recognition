# Handwritten Letter Recognition (EMNIST, CNN)

A convolutional neural network that recognises handwritten letters A–Z, built with careful data cleaning and systematic model improvement.

## Results

| Model | Validation accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|
| Baseline CNN | 92.7% | — | — |
| **Final regularised CNN (fine-tuned)** | — | **94.9%** | **0.949** |

- The biggest gains came from **cleaning the data** (removing 10K+ invalid labels and duplicates) and from **regularisation** (batch normalisation and dropout).
- Error analysis showed that most mistakes are between letters that look alike when handwritten, such as **I / L** and **G / Q**.
- The notebook ends with a prediction demo on individual images.

## Dataset

**EMNIST Letters**: 28×28 greyscale images of handwritten letters, about **99K** images after loading.
The CSV (`emnist-letters.csv`) is not included because of its size. It is available from the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) (also mirrored on Kaggle). Place it next to the notebook to rerun.

## Approach

1. **Data cleaning**: removed 10K+ invalid labels and duplicate images, and corrected EMNIST's rotated or flipped image orientation.
2. **Fair split**: stratified 70 / 15 / 15 train / validation / test split.
3. **Baseline CNN**: convolution, pooling and dense layers as a reference point.
4. **Improvement experiments**: batch normalisation, dropout, augmentation tests and fine-tuning, each compared on the validation set.
5. **Final evaluation**: test accuracy, macro F1, per-class report and confusion matrix.

## Files

| File | Description |
|---|---|
| `emnist_letter_recognition.ipynb` | Full workflow with outputs and charts |
| `partA_best_final_model.weights.h5` | Trained weights of the final model |

## Tools

Python · TensorFlow / Keras · NumPy · pandas · scikit-learn · Matplotlib

## Run it

```bash
pip install -r requirements.txt
jupyter notebook emnist_letter_recognition.ipynb
```

## Author

**Phyu Sin Lynn (Emi)** — Diploma in Applied AI & Analytics, Singapore Polytechnic
[Portfolio](https://emi-psl.github.io/portfolio/) · [GitHub](https://github.com/emi-psl) · [LinkedIn](https://www.linkedin.com/in/phyusinlynn)

*Completed as coursework for the Deep Learning module at Singapore Polytechnic.*
