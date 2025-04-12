# REMEMBER: Retrieval-based Explainable Multimodal Evidence-guided Modeling for Brain Evaluation and Reasoning

This repository provides the official implementation of **REMEMBER**, a retrieval-augmented, explainable framework for zero- and few-shot diagnosis of neurodegenerative diseases from brain MRI. REMEMBER simulates clinical reasoning by aligning input images with textual medical knowledge, retrieving similar annotated reference cases, and producing structured, reference-backed explanations.

## 🚀 Key Features

- ✅ **Zero- and Few-Shot Generalization**: Perform diagnostic prediction without large-scale labeled data.
- 📖 **Clinically Aligned Explanations**: Outputs reference-based justification for each prediction.
- 🔍 **Multimodal Reasoning**: Leverages both radiology images and associated text (e.g., abnormality descriptions).
- 🎯 **High Accuracy and Interpretability**: Validated across four neurodegenerative diagnosis tasks.

## 📁 Directory Structure

```bash
.
├── zero-shot/        # Zero-shot inference notebooks
├── attention/        # Attention-based inference notebooks with evidence integration
└── requirements.txt  # Python package dependencies
```

Each of the `zero-shot/` and `attention/` folders includes five Jupyter notebooks:

| File | Task |
|------|------|
| `b-1-abnormality-type-prediction.ipynb` | Abnormality type classification |
| `b-2-binary-dementia-classification-on-mindset.ipynb` | Binary classification on MINDSet |
| `b-2-binary-dementia-classification-on-public.ipynb` | Binary classification on public dataset |
| `b-3-dementia-type-classification.ipynb` | Multi-class dementia subtype classification |
| `b-4-dementia-severity-classification.ipynb` | Dementia severity staging |

---

## 🧠 Diagnostic Tasks

REMEMBER supports evaluation on the following clinical tasks:

| Task | Labels |
|------|--------|
| **Abnormality Type** | Normal, MTL Atrophy, WMH, Other Atrophy |
| **Binary Dementia Classification** | Demented vs Non-demented |
| **Dementia Type Classification** | Non-demented, Alzheimer's Disease, Other Dementia |
| **Dementia Severity Staging** | Non-demented, Very Mild, Mild, Moderate |

---

## 🛠️ Installation & Dependencies

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

See the notebooks for task-specific requirements.

---

## 💻 Usage Instructions

All experiments can be executed directly via Jupyter notebooks.

### Running Zero-shot Inference

```bash
cd zero-shot
# Launch the desired notebook, e.g.:
# jupyter notebook b-1-abnormality-type-prediction.ipynb
```

### Running Attention-Guided Inference

```bash
cd attention
# Launch the corresponding attention-based notebook
# jupyter notebook b-1-abnormality-type-prediction.ipynb
```

Each notebook will:
- Load the pretrained REMEMBER model and necessary embeddings
- Load MINDSet and/or public test images
- Perform zero- or few-shot prediction
- Retrieve top-k reference cases
- Output evaluation metrics

---

## 📊 Output Format

Each notebook produces:

- **Prediction outputs**: Class labels and confidence scores
- **Performance Metrics**: Accuracy, F1, Precision, Recall, Specificity (macro-averaged)

---

## 📁 Datasets

- **MINDSet**: A curated dataset of 170 annotated MRI images paired with radiology-style text. Used for evidence retrieval and few-shot training.
- **Public Dataset**: A 2D axial slice dataset for AD staging and binary classification.

Refer to the paper for details.

---

## 🧪 Model Checkpoints

The pretrained REMEMBER model weights will be released on Hugging Face upon publication:

```
https://huggingface.co/anonymous/REMEMBER
```

---

## 🔒 Notes on Anonymity

This repository is shared as part of a **double-blind conference submission**. Author and affiliation information is withheld to preserve anonymity.

---

## 📜 License

This code is released for **non-commercial, academic research use only**. Redistribution or use for clinical deployment is prohibited without further agreement.

---

## 💬 Citation

To be updated upon publication.
