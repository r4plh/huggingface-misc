# Python script for a casual language model
from datasets import Dataset
from collections import defaultdict
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
from datasets import load_dataset , DatasetDict , load_from_disk
from transformers import AutoTokenizer
import json

def key_in_string(keywords , py_scripts):
    for keyword in keywords:
        if keyword in py_scripts:
            return True
    return False     

# When you use the streaming=True option with the load_dataset function from the Hugging Face datasets library, the dataset object becomes an iterable dataset. This means you can iterate over the dataset entries one-by-one as they are streamed from the source, without loading the entire dataset into memory. This approach is particularly useful for processing large datasets efficiently in environments with limited memory.  

def making_dataset(data , filters):
    total = 1
    filtered_dict = defaultdict(list)
    for sample in tqdm(data):
        total = total + 1
        if key_in_string(filters , sample["content"]):
            for k,v in sample.items():
                filtered_dict[k].append(v)

    print(f"The filtered dataset is {(len(filtered_dict)/total)*100} of the original dataset")
    return Dataset.from_dict(filtered_dict)

filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

# data = load_dataset("transformersbook/codeparrot-train", split="train",streaming=True)
# filtered_data = making_dataset(data , filters)

# Load the datasets in streaming mode
# ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train", streaming=True)
# ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation", streaming=True)

# # Create a DatasetDict to organize the streamed datasets
# raw_datasets = DatasetDict(
#     {
#         "train": ds_train,
#         "valid": ds_valid,
#     }
# )

# # Since the dataset is streamed, the actual type might be different
# print(type(raw_datasets['train']))
# # ---> <class 'datasets.iterable_dataset.IterableDataset'>

# print(type(next(iter(raw_datasets['train']))))
# # ---> <class 'dict'>

# train_dict = defaultdict(list)

# total = 1
# for sample in raw_datasets['train']:
#     total = total + 1
#     if (total%1000 == 0):
#         print(f"{total} rows have been added") 
#     for k,v in sample.items():
#         train_dict[k].append(v)

# with open('json_data.json','w') as f:
#     json.dump(train_dict,f)

full_data = load_dataset("json",data_files="/Users/0xr4plh/Documents/huggingface-misc/PEFT/codeparrot-ds-train.json")

val_data = load_dataset("json",data_files="/Users/0xr4plh/Documents/huggingface-misc/PEFT/codeparrot-ds-valid.json")

# # ---> DatasetDict({
#     train: Dataset({
#         features: ['repo_name', 'path', 'copies', 'size', 'content', 'license'],
#         num_rows: 606720
#     })
# })


# print(len(full_data["train"][:2]["content"]))


context_length = 512
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

def tokenize(element):
    output = tokenizer(element["content"] , return_overflowing_tokens=True , max_length=512 , truncation=True , return_length=True)

    input_data = []

    for input_id in output["input_ids"]:
        if len(input_id) == 512:
            input_data.append(input_id)

    return {"input_ids" : input_data}        

# tokenized_dataset = full_data.map(tokenize , batched=True, remove_columns=full_data["train"].column_names)

# tokenized_dataset_val = val_data.map(tokenize , batched=True, remove_columns=full_data["train"].column_names)

# tokenized_dataset_val.save_to_disk("/Users/0xr4plh/Documents/huggingface-misc/PEFT/val_dataset")

# print(tokenized_dataset_val)
# DatasetDict({
#     train: Dataset({
#         features: ['input_ids'],
#         num_rows: 22000
#     })
# })


# tokenized_dataset.save_to_disk('/Users/0xr4plh/Documents/huggingface-misc/PEFT/dataset')

tokenized_dataset = load_from_disk("/Users/0xr4plh/Documents/huggingface-misc/PEFT/train_dataset")
tokenized_dataset_val = load_from_disk("/Users/0xr4plh/Documents/huggingface-misc/PEFT/val_dataset")

# print(tokenized_dataset)

# print(len(tokenizer))
# # ---> 50000

# only the config is same that is skeleton is same as gpt2 that means the transformers layers and all other architecture is same , the weights and biases are not from pretrained , that we are only going to do from scratch , from fixing the skeleton we will also get to know the number of trainable parameters

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config) # model size = 124.2 million parameters of gpt-2 skeleton , we have to put in right weights after we do training from scratch on our task

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) # ---> 3 things it gives as outputs - input_ids , attention_mask , labels

# Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=False,  # Set this to False or remove this line entirely
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset_val["train"] # This is because the name of the val split is also named as train split only
)

trainer.train()

trainer.push_to_hub()
