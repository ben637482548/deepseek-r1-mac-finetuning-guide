# Ultimate Guide: Fine-tuning DeepSeek-R1 (Distilled Llama) Locally on M-series Mac (part A)
*A comprehensive, step-by-step guide with validation steps and troubleshooting*

> This guide is an expanded version of the original tutorial by [Avi Chawla](https://github.com/ChawlaAvi). The original tutorial can be found [here](https://x.com/_avichawla/status/1884126766132011149).

## Section 0: Prerequisites
### System Requirements
- macOS ≥ 13.0 (Ventura or newer)
- M1/M2/M3 Mac with ≥16GB RAM
- At least 20GB free disk space
- Xcode Command Line Tools
- Active internet connection for initial downloads

### Initial Setup Commands
```bash
# Install Xcode CLI tools if not already installed
xcode-select --install

# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Miniforge3 for ARM64
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

## Section 1: Environment Setup
### Create Python Environment
```bash
# Create new conda environment
conda create -n deepseek python=3.11
conda activate deepseek

# Install PyTorch with Metal support
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install UnslothAI with Apple Silicon support
pip install "unsloth[apple-m1]"

# Install Ollama
brew install ollama
```

### Validate Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Should output True for both MPS checks
```

## Section 2: Load Model
### Initialize Model and Tokenizer
```python
from unsloth import FastLanguageModel
import torch

# Set environment variables for Metal optimization
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.95'

# Define model path
MODEL = "unsloth/DeepSeek-R1-Distill-llama-8B-unsloth-bnb-4bit"

# Load model with Metal acceleration
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,  # Enable 4-bit quantization
)

# Move model to MPS device
device = torch.device("mps")
model = model.to(device)
```

### Validation Check
```python
# Test tokenizer and model
test_input = "Hello, world!"
tokens = tokenizer(test_input, return_tensors="pt").to(device)
print(f"Tokenized output: {tokens}")
```

## Section 3: Configure LoRA
### Define LoRA Parameters
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=4,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_rslora=False,
    loftq_config=None
)
```

## Section 4: Prepare Dataset
### Load and Process Dataset
```python
from datasets import load_dataset
from unsloth import to_sharegpt, standardize_sharegpt

# Load Alpaca dataset
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

# Convert to ShareGPT format
dataset = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[\nYour input is:\n{input}]",
    output_column_name="output",
    conversation_extension=3
)

# Standardize format
dataset = standardize_sharegpt(dataset)
```

### Dataset Validation
```python
# Verify dataset format
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample entry:\n{sample}")

# Check token lengths
max_length = max(len(tokenizer.encode(str(x))) for x in dataset)
print(f"Maximum sequence length: {max_length}")
```

## Section 5: Configure Training
### Define Training Arguments
```python
from transformers import TrainingArguments
from ttl import SFTTrainer

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=60,
    learning_rate=2e-4,
    optim="adamw_8bit",
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision
    output_dir="./results",
    save_strategy="steps",
    save_steps=20,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args
)
```

## Section 6: Training Process
### Start Training
```python
# Enable Metal performance monitoring
print("Starting training...")
trainer_stats = trainer.train()

# Monitor progress
print(f"Initial loss: {trainer_stats.training_loss[0]}")
print(f"Final loss: {trainer_stats.training_loss[-1]}")
```

### Training Monitoring Guidance
- Expected loss pattern: 1.8-2.0 initially, decreasing to 1.0-1.3
- Monitor Activity Monitor for GPU usage
- Watch for memory warnings in terminal
- Training should take approximately 30-60 minutes

## Section 7: Export to Ollama
### Save Model
```python
# Save model in GGUF format
model.save_pretrained_gguf("deepseek_finetuned", tokenizer)

# Create Modelfile
with open("Modelfile", "w") as f:
    f.write("""FROM deepseek-r1:8b-base
PARAMETER temperature 0.7
PARAMETER top_p 0.7
PARAMETER stop "User:"
PARAMETER stop "Assistant:"
LICENSE Apache 2.0
TEMPLATE """{{.System}}
User: {{.Prompt}}
Assistant: """
""")

# Create Ollama model
!ollama create deepseek_finetuned -f ./Modelfile
```

## Section 8: Test and Validate
### Model Testing
```python
import ollama

# Test the model
response = ollama.chat(model='deepseek_finetuned', messages=[
    {
        'role': 'user',
        'content': 'What is 2+2?'
    }
])
print(response['message']['content'])
```

## Troubleshooting Guide
### Common Issues and Solutions
1. Metal Device Errors
   ```python
   # Reset Metal device
   torch.mps.empty_cache()
   ```

2. Memory Issues
   - Reduce batch size
   - Enable gradient checkpointing
   - Clear Python memory:
     ```python
     import gc
     gc.collect()
     ```

3. Training Crashes
   - Verify macOS version ≥13.0
   - Check available memory
   - Reduce model size or use more quantization

### Performance Optimization
- Set environment variables:
  ```bash
  export MPS_GRAPH_COMPILE_SYNCHRONOUS=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
  ```


# Part B: Comprehensive Error Resolution Guide for DeepSeek-R1 Fine-tuning
*A companion guide for troubleshooting every step of the fine-tuning process*

## Section 0: Prerequisites Issues
### Xcode Installation Problems
```bash
Error: xcode-select: error: command line tools are already installed
```
**Solution:**
```bash
# Remove existing installation
sudo rm -rf /Library/Developer/CommandLineTools
# Reinstall
xcode-select --install
```

### Homebrew Issues
```bash
Error: Permission denied @ dir_s_mkdir - /usr/local/Cellar
```
**Solution:**
```bash
# Fix permissions
sudo chown -R $(whoami) /usr/local/*
# Retry installation
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Miniforge Installation Errors
```bash
Error: No space left on device
```
**Solution:**
1. Clean up unnecessary files:
```bash
brew cleanup
conda clean --all
```
2. Check space requirements:
```bash
df -h
```
3. Required space: at least 20GB free

## Section 1: Environment Setup Issues
### PyTorch Installation Problems
```bash
ERROR: Could not find a version that satisfies the requirement torch
```
**Solution:**
```bash
# Clear pip cache
pip cache purge
# Try alternative installation
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

### UnslothAI Installation Errors
```bash
Error: Failed building wheel for unsloth
```
**Solution:**
1. Install build dependencies:
```bash
conda install -y cmake ninja
pip install --upgrade pip setuptools wheel
```
2. Try alternative installation:
```bash
pip install "unsloth[apple-m1] @ git+https://github.com/unslothai/unsloth.git"
```

### Metal Acceleration Issues
```python
RuntimeError: MPS backend not available
```
**Solution:**
1. Verify macOS version:
```bash
sw_vers
# Must be ≥13.0
```
2. Check Metal support:
```python
import torch
print(torch.backends.mps.is_built())
if not torch.backends.mps.is_available():
    print("Update macOS to version 13.0 or later")
```

## Section 2: Model Loading Issues
### Out of Memory Errors
```python
RuntimeError: out of memory
```
**Solution:**
1. Enable memory optimization:
```python
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
torch.mps.empty_cache()
```
2. Reduce model size:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=1024,  # Reduced from 2048
    load_in_8bit=True,    # Use 8-bit instead of 4-bit
)
```

### Tokenizer Issues
```python
OSError: Can't load tokenizer for 'unsloth/DeepSeek-R1'
```
**Solution:**
```python
# Force download tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
    use_fast=False
)
```

## Section 3: LoRA Configuration Problems
### CUDA Error Messages
```python
RuntimeError: CUDA error: no kernel image is available for execution
```
**Solution:**
```python
# Force Metal backend
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device("mps")
```

### Gradient Checkpointing Errors
```python
ValueError: Gradient checkpointing is not compatible
```
**Solution:**
```python
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing=False,  # Disable if problematic
    lora_alpha=16,
    lora_dropout=0.1,    # Add dropout for stability
)
```

## Section 4: Dataset Preparation Issues
### Dataset Loading Failures
```python
FileNotFoundError: Dataset vicgalle/alpaca-gpt4 not found
```
**Solution:**
1. Check internet connection
2. Try alternative dataset source:
```python
# Alternative loading method
dataset = load_dataset(
    "json", 
    data_files={"train": "path/to/local/alpaca.json"},
    split="train"
)
```

### Format Conversion Errors
```python
KeyError: instruction not found in dataset
```
**Solution:**
```python
# Verify dataset structure
print(dataset[0].keys())
# Map correct fields
dataset = dataset.map(
    lambda x: {
        "instruction": x.get("prompt", ""),
        "input": x.get("context", ""),
        "output": x.get("response", "")
    }
)
```

## Section 5: Training Configuration Issues
### GPU Memory Allocation Errors
```python
RuntimeError: MPS backend: Metal out of memory
```
**Solution:**
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,     # Reduce batch size
    gradient_accumulation_steps=8,      # Increase steps
    fp16=False,                        # Disable mixed precision
    optim="adamw_torch",               # Use standard optimizer
)
```

### Loss Explosion/NaN Issues
```python
WARNING: Loss is NaN
```
**Solution:**
```python
training_args = TrainingArguments(
    learning_rate=1e-4,                # Reduce learning rate
    max_grad_norm=1.0,                 # Add gradient clipping
    warmup_steps=100,                  # Add warmup
)
```

## Section 6: Training Process Issues
### Training Stuck/No Progress
**Symptoms:**
- Loss not decreasing
- GPU utilization low
- Training seems frozen

**Solution:**
1. Check Progress:
```python
# Monitor training metrics
print(f"Step: {trainer.state.global_step}")
print(f"Loss: {trainer.state.log_history[-1]}")
```

2. Reset Training:
```python
# Clear cache
torch.mps.empty_cache()
import gc
gc.collect()

# Restart training with monitoring
trainer.train(
    resume_from_checkpoint=False,
    report_to="tensorboard"
)
```

### Checkpoint Saving Errors
```python
OSError: Can't save model
```
**Solution:**
```python
# Set explicit save directory with permissions
import os
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)
os.chmod(save_dir, 0o777)

training_args = TrainingArguments(
    output_dir=save_dir,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,  # Keep only last 2 checkpoints
)
```

## Section 7: Ollama Export Issues
### GGUF Conversion Errors
```bash
Error: Failed to convert model to GGUF format
```
**Solution:**
1. Check disk space
2. Try alternative conversion:
```python
# Save in safetensors format first
model.save_pretrained("./model_safetensors", safe_serialization=True)

# Then convert using llama.cpp
!./llama.cpp/convert.py ./model_safetensors --outfile model.gguf
```

### Ollama Model Creation Fails
```bash
Error: failed to create model
```
**Solution:**
1. Check Modelfile syntax:
```bash
# Validate Modelfile
ollama show deepseek_finetuned
```

2. Clean and retry:
```bash
# Remove existing model
ollama rm deepseek_finetuned
# Clear cache
rm -rf ~/.ollama/models/deepseek_finetuned
# Retry creation
ollama create deepseek_finetuned -f ./Modelfile
```

## Section 8: Testing Issues
### Model Response Problems
**Symptoms:**
- No output
- Garbage output
- Very slow responses

**Solution:**
1. Check model loading:
```python
# Verify model is loaded correctly
response = ollama.list()
print(response)
```

2. Adjust inference parameters:
```python
response = ollama.chat(
    model='deepseek_finetuned',
    messages=[{'role': 'user', 'content': 'Test prompt'}],
    options={
        'temperature': 0.7,
        'top_p': 0.9,
        'num_predict': 100,
        'stop': ['User:', 'Assistant:']
    }
)
```

### System Resource Monitoring
Always monitor system resources during the entire process:
```bash
# Monitor GPU
sudo powermetrics --samplers gpu_power -i 1000

# Monitor memory
top -l 1 -n 0 -s 0 | grep PhysMem

# Monitor disk space
df -h
```
