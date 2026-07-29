# Prompt Tuning Gemma for Insurance Question Answering

## Overview

This project demonstrates **prompt tuning** of **Google Gemma 1.1 7B Instruct** using the **PEFT (Parameter-Efficient Fine-Tuning)** library for insurance question answering. The notebook walks through the complete workflow from loading a pretrained model to fine-tuning, evaluation, and inference.

Rather than updating all model parameters, prompt tuning learns a small set of virtual prompt embeddings while keeping the base model frozen, making training significantly more efficient.

---

## Features

* Load and run inference with the pretrained Gemma 1.1 7B Instruct model
* Compare responses with and without prompt engineering
* Prepare and preprocess the InsuranceQA dataset
* Visualize token length distributions
* Fine-tune using Prompt Tuning (PEFT)
* Save and reload trained adapters
* Evaluate the fine-tuned model
* Export inference results for further analysis
* Optional upload of trained adapters to Hugging Face Hub

---

## Model

* **Base Model:** `google/gemma-1.1-7b-it`
* **Fine-Tuning Method:** Prompt Tuning (PEFT)
* **Task:** Causal Language Modeling

---

## Dataset

The notebook uses insurance question-answering datasets from Hugging Face, including:

* `deccan-ai/insuranceQA-v2`
* `SJSU-SP24-DATA298-T3/insurance_dataset`

Each sample contains an insurance-related question and an expected professional answer used for supervised prompt tuning.

---

## Workflow

1. Install dependencies
2. Load Gemma model and tokenizer
3. Run baseline inference
4. Create instruction prompts
5. Prepare and tokenize the dataset
6. Visualize token distributions
7. Configure Prompt Tuning
8. Fine-tune the model
9. Save trained prompt adapters
10. Reload adapters for inference
11. Evaluate on the test set
12. Export predictions to CSV

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* PEFT
* Datasets
* Accelerate
* Matplotlib
* Pandas
* NumPy

---

## Results

The notebook compares:

* Base Gemma model
* Prompt-engineered inference
* Prompt-tuned Gemma model

The generated responses are exported for downstream evaluation and comparison.

---

## Repository Structure

```
.
├── prompt_tuning_gemma_insurance_clean.ipynb
├── README.md
└── generated_outputs/
```

---

## Running the Notebook

Install the required packages:

```bash
pip install torch transformers datasets peft accelerate matplotlib pandas numpy
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```
prompt_tuning_gemma_insurance_clean.ipynb
```

---

## Future Improvements

* Compare Prompt Tuning with LoRA and QLoRA
* Evaluate on additional insurance QA benchmarks
* Integrate automatic evaluation metrics (BLEU, ROUGE, BERTScore)
* Deploy the fine-tuned model as an inference API
* Experiment with larger Gemma models

---

## License

This project is intended for educational and research purposes.

