from transformers import RobertaConfig, BertTokenizerFast
from datasets import load_dataset, load_from_disk
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import os 
from accelerate import Accelerator
# only use 6 and 7


# Checking if CUDA is available
cuda_available = torch.cuda.is_available()
print(cuda_available, flush=True)

# bert-base
config = RobertaConfig(
    vocab_size=51500,
    max_position_embeddings=520,
)

accelerator = Accelerator()

# Load the pretrained tokenizer
tokenizer = BertTokenizerFast.from_pretrained("/home/junjdong/TrainMorganBERT2/MorganBERT_base_full_r_1_s_0_radiusFirst_f_300/vocab", truncation=True, max_len=512, return_tensors="pt")

model = RobertaForMaskedLM(config=config)

dataset = load_dataset('text', data_files='/home/junjdong/TrainMorganBERT2/corpus_sentence_pretraining_full_r_1_s_0_radiusFirst.txt', cache_dir='/home/junjdong/TrainMorganBERT2/cache')



def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# use multiprocessing to speed up the tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=256, remove_columns=["text"]) # type: ignore

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
print("training start")
training_args = TrainingArguments(
    output_dir="/home/junjdong/TrainMorganBERT2/MorganBERT_base_full_r_1_s_0_radiusFirst_f_300",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
)


# Train with the accelerator
trainer = accelerator.prepare(Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    # use the entire tokenized dataset
    train_dataset=tokenized_dataset['train'],
))

# Enable accelerator
# trainer = accelerator.prepare(trainer)

print('training start', flush=True)
trainer.train()

unwrapped_model = accelerator.unwrap_model(trainer.model)
unwrapped_model.save_pretrained("/home/junjdong/TrainMorganBERT2/MorganBERT_base_full_r_1_s_0_radiusFirst_f_300",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)
