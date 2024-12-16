## Iterable Dataset (Streaming Mode)

### **Introduction**
Using `streaming=True` with the `load_dataset()` function from the Hugging Face `datasets` library transforms the dataset into a generator-like object. This approach is particularly useful for managing large datasets efficiently.

### **Advantages**
- **Generator Functions**: Yields data items sequentially, facilitating the processing of each item without loading the entire dataset into memory.
- **Memory Efficiency**: Loads only the necessary data for each iteration, avoiding the need to fit the entire dataset into RAM.
- **Handling Large Datasets**: Ideal for datasets that are too large to be loaded into memory, such as those in the range of hundreds of gigabytes or more.
- **Use Cases**: Best for processing data directly streamed from remote servers or in environments with limited memory resources.

## Normal Dataset (In-Memory)

### **Introduction**
Loading a dataset normally without `streaming=True` stores the entire data split in memory as a `Dataset` object, which is straightforward but memory-intensive.

### **Advantages**
- **In-Memory Access**: Immediate access to all data, which is necessary for operations that require random access to various dataset parts.
- **Fast Access**: Provides quicker data retrieval since the entire dataset is pre-loaded into memory.
- **Use Cases**: Suitable for smaller datasets that can comfortably fit into memory or when extensive data manipulation is required.

## Methods to Create Iterable Datasets

### **1. Using Python Generators**
Create a generator function that yields data items one at a time, allowing for memory-efficient data loading from diverse sources.

### **2. Streaming with Hugging Face `datasets`**
Utilize the `streaming=True` option in the `load_dataset` function to stream datasets and process them sequentially from remote sources.

### **3. Direct Conversion to Iterable Dataset**
Direct Conversion to Iterable Dataset to_iterable_dataset() method in Hugging Face datasets

# Building a Causal Language Model (CLM) From Scratch: My Approach

## 1. Choosing the Model Based on Compute
The first step is to consider the computational resources I have. If I’m working locally without powerful GPUs, I might go for a smaller model like GPT-2 small or medium. If I have more compute, I can pick a larger model. The choice of model size depends on my available hardware.

## 2. Training From Scratch vs. Fine-Tuning
Since the requirement is to train from scratch, I’m not tied to an existing model or tokenizer. Typically, when fine-tuning a pre-trained model, I must use its original tokenizer to maintain consistency. But starting from scratch means I can build or choose a tokenizer freely, as long as its vocabulary size matches the model’s vocabulary. This ensures that both the model and the tokenizer understand tokens in the same way.

## 3. Using an Existing Tokenizer if Possible
If the dataset I’m given comes from a domain where good pre-trained tokenizers are already available—like code, biology, or medical text—it’s more efficient to use one of those. There’s no need to reinvent the wheel if a high-quality tokenizer is out there. For example, if my dataset is code-related, I can choose a tokenizer trained on a large code corpus. I’d then initialize a generative model (e.g., GPT-2) and set its vocabulary size according to that tokenizer. From there, I’d prepare the training data for the CLM task. With a proper data collator, label creation is straightforward—I only need to decide on the context window size and do any necessary data preprocessing.

## 4. Training a New Tokenizer for New Domains
If my dataset is from a domain that lacks a suitable tokenizer—say a low-resource language or a very specialized text type—I’d train a new tokenizer. I could use Byte Pair Encoding (BPE), WordPiece, or another algorithm. After training it on the dataset, I’d get the vocabulary size and feed that into my model configuration. Using the newly created tokenizer, I’d prepare the data, rely on the data collator to handle labeling, and write my training script to run the process from scratch.

## 5. Understanding the Dataset
Finally, I must analyze the dataset itself. What is its domain, size, and complexity? This affects whether I use an existing tokenizer or build a new one, how I set my context window, and what preprocessing steps are needed. Understanding the dataset is key to making informed decisions about the entire setup.

---

**In summary**, I’ll pick a model size that fits my compute, decide on using an existing tokenizer or training a new one depending on the domain, and then prepare the data and run the training script. Training from scratch gives me complete freedom in choosing or building the tokenizer and model architecture to ensure they align perfectly.

