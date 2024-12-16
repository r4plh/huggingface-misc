# Python script for a casual language model
from datasets import Dataset
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset

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
data = load_dataset("transformersbook/codeparrot-train", split="train",streaming=True)

filtered_data = making_dataset(data , filters)



