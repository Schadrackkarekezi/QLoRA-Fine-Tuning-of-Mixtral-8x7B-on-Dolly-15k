Mixtral-8x7B QLoRA Fine-Tuning

This repository contains a complete code pipeline for fine-tuning Mixtral-8x7B-Instruct using QLoRA on the Databricks Dolly 15k dataset.
All training is done in 4-bit quantization, with LoRA adapters applied to reduce compute and memory cost.

📁 Repository Structure
├── mixtral_qlora.ipynb        # Main fine-tuning code (QLoRA + Dolly 15k)
├── finetune_comparison.py     # Perplexity evaluation code
├── assets/
│   └── train_loss.png         # Training loss curve
└── README.md

⚙️ Installation
pip install transformers peft datasets accelerate bitsandbytes


For GPUs supporting bfloat16 and 4-bit quantization.

🧠 Fine-Tuning (Notebook)

Run the main training notebook:

jupyter notebook mixtral_qlora.ipynb


Inside the notebook, the following steps are performed:

# Load Dolly dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Load Mixtral-8x7B in 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# Apply LoRA
lora_config = LoraConfig(r=64, lora_alpha=16, target_modules=['q_proj','v_proj'])
model = get_peft_model(model, lora_config)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
trainer.train()

📊 Evaluate Perplexity

Use:

python finetune_comparison.py


Example from the script:

ppl_base = calculate_perplexity(base_model, tokenizer, val_dataset)
ppl_adapter = calculate_perplexity(model_with_adapter, tokenizer, val_dataset)
print(ppl_base, ppl_adapter)

📉 Training Loss Curve

🚀 Summary of Improvements

Trained only ~2% of parameters using LoRA

Reduced memory cost via 4-bit quantization

Improved perplexity compared to base Mixtral

Stable convergence during training
