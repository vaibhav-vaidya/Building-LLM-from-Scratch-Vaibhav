# Databricks notebook source
# MAGIC %md
# MAGIC | Author  | VAIBHAV SHASHIKANT VAIDYA |
# MAGIC |---------|---------------------------|
# MAGIC | Date    | 21-12-2025                |
# MAGIC | Version | V1.0                      |
# MAGIC | Topic   | Text Split and Tokenization |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading in a short story as text sample into Python.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Creating Tokens

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The print command prints the total number of characters followed by the first 100
# MAGIC characters of this file for illustration purposes. </div>

# COMMAND ----------

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Our goal is to tokenize this 20,479-character short story into individual words and special
# MAGIC characters that we can then turn into embeddings for LLM training  </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Note that it's common to process millions of articles and hundreds of thousands of
# MAGIC books -- many gigabytes of text -- when working with LLMs. However, for educational
# MAGIC purposes, it's sufficient to work with smaller text samples like a single book to
# MAGIC illustrate the main ideas behind the text processing steps and to make it possible to
# MAGIC run it in reasonable time on consumer hardware. </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC How can we best split this text to obtain a list of tokens? For this, we go on a small
# MAGIC excursion and use Python's regular expression library re for illustration purposes. (Note
# MAGIC that you don't have to learn or memorize any regular expression syntax since we will
# MAGIC transition to a pre-built tokenizer later in this chapter.) </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Using some simple example text, we can use the re.split command with the following
# MAGIC syntax to split a text on whitespace characters:</div>

# COMMAND ----------

import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC The result is a list of individual words, whitespaces, and punctuation characters:
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Let's modify the regular expression splits on whitespaces (\s) and commas, and periods
# MAGIC ([,.]):</div>

# COMMAND ----------

result = re.split(r'([,.]|\s)', text)

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC We can see that the words and punctuation characters are now separate list entries just as
# MAGIC we wanted
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC A small remaining issue is that the list still includes whitespace characters. Optionally, we
# MAGIC can remove these redundant characters safely as follows:</div>

# COMMAND ----------

result = [item for item in result if item.strip()]
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC REMOVING WHITESPACES OR NOT
# MAGIC
# MAGIC
# MAGIC When developing a simple tokenizer, whether we should encode whitespaces as
# MAGIC separate characters or just remove them depends on our application and its
# MAGIC requirements. Removing whitespaces reduces the memory and computing
# MAGIC requirements. However, keeping whitespaces can be useful if we train models that
# MAGIC are sensitive to the exact structure of the text (for example, Python code, which is
# MAGIC sensitive to indentation and spacing). Here, we remove whitespaces for simplicity
# MAGIC and brevity of the tokenized outputs. Later, we will switch to a tokenization scheme
# MAGIC that includes whitespaces.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The tokenization scheme we devised above works well on the simple sample text. Let's
# MAGIC modify it a bit further so that it can also handle other types of punctuation, such as
# MAGIC question marks, quotation marks, and the double-dashes we have seen earlier in the first
# MAGIC 100 characters of Edith Wharton's short story, along with additional special characters: </div>

# COMMAND ----------

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# COMMAND ----------

# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]
print(result)

# COMMAND ----------

text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Now that we got a basic tokenizer working, let's apply it to Edith Wharton's entire short
# MAGIC story:
# MAGIC
# MAGIC </div>

# COMMAND ----------

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

# COMMAND ----------

print(len(preprocessed))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Creating Token IDs

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In the previous section, we tokenized Edith Wharton's short story and assigned it to a
# MAGIC Python variable called preprocessed. Let's now create a list of all unique tokens and sort
# MAGIC them alphabetically to determine the vocabulary size:</div>

# COMMAND ----------

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
 
print(vocab_size)

# COMMAND ----------

print(all_words)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC After determining that the vocabulary size is 1,130 via the above code, we create the
# MAGIC vocabulary and print its first 51 entries for illustration purposes:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# vocab = {}
# for integer, token in enumerate(all_words):
#     vocab[token] = integer

# COMMAND ----------

vocab = {token:integer for integer,token in enumerate(all_words)}

# COMMAND ----------

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC As we can see, based on the output above, the dictionary contains individual tokens
# MAGIC associated with unique integer labels. 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Later in this book, when we want to convert the outputs of an LLM from numbers back into
# MAGIC text, we also need a way to turn token IDs into text. 
# MAGIC
# MAGIC For this, we can create an inverse
# MAGIC version of the vocabulary that maps token IDs back to corresponding text tokens.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's implement a complete tokenizer class in Python.
# MAGIC
# MAGIC The class will have an encode method that splits
# MAGIC text into tokens and carries out the string-to-integer mapping to produce token IDs via the
# MAGIC vocabulary. 
# MAGIC
# MAGIC In addition, we implement a decode method that carries out the reverse
# MAGIC integer-to-string mapping to convert the token IDs back into text.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Step 1: Store the vocabulary as a class attribute for access in the encode and decode methods
# MAGIC     
# MAGIC Step 2: Create an inverse vocabulary that maps token IDs back to the original text tokens
# MAGIC
# MAGIC Step 3: Process input text into token IDs
# MAGIC
# MAGIC Step 4: Convert token IDs back into text
# MAGIC
# MAGIC Step 5: Replace spaces before the specified punctuation
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC

# COMMAND ----------

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's instantiate a new tokenizer object from the SimpleTokenizerV1 class and tokenize a
# MAGIC passage from Edith Wharton's short story to try it out in practice:
# MAGIC </div>

# COMMAND ----------

tokenizer = SimpleTokenizerV1(vocab)  #CREATING AN INSTANCE OF THE CLASS

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC The code above prints the following token IDs:
# MAGIC Next, let's see if we can turn these token IDs back into text using the decode method:
# MAGIC </div>

# COMMAND ----------

tokenizer.decode(ids)


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Based on the output above, we can see that the decode method successfully converted the
# MAGIC token IDs back into the original text.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC So far, so good. We implemented a tokenizer capable of tokenizing and de-tokenizing
# MAGIC text based on a snippet from the training set. 
# MAGIC
# MAGIC Let's now apply it to a new text sample that
# MAGIC is not contained in the training set:
# MAGIC </div>

# COMMAND ----------

text = "Hello, do you like tea?"
print(tokenizer.encode(text))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC The problem is that the word "Hello" was not used in the The Verdict short story. 
# MAGIC
# MAGIC Hence, it
# MAGIC is not contained in the vocabulary. 
# MAGIC
# MAGIC This highlights the need to consider large and diverse
# MAGIC training sets to extend the vocabulary when working on LLMs.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### ADDING SPECIAL CONTEXT TOKENS (ChatGPT also uses the same)
# MAGIC
# MAGIC In the previous section, we implemented a simple tokenizer and applied it to a passage
# MAGIC from the training set. 
# MAGIC
# MAGIC In this section, we will modify this tokenizer to handle unknown
# MAGIC words.
# MAGIC
# MAGIC
# MAGIC In particular, we will modify the vocabulary and tokenizer we implemented in the
# MAGIC previous section, SimpleTokenizerV2, to support two new tokens, <|unk|> and
# MAGIC <|endoftext|>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC We can modify the tokenizer to use an <|unk|> token if it
# MAGIC encounters a word that is not part of the vocabulary. 
# MAGIC
# MAGIC Furthermore, we add a token between
# MAGIC unrelated texts. 
# MAGIC
# MAGIC For example, when training GPT-like LLMs on multiple independent
# MAGIC documents or books, it is common to insert a token before each document or book that
# MAGIC follows a previous text source
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's now modify the vocabulary to include these two special tokens, <unk> and
# MAGIC <|endoftext|>, by adding these to the list of all unique words that we created in the
# MAGIC previous section:
# MAGIC </div>

# COMMAND ----------

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

# COMMAND ----------

len(vocab.items())


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Based on the output of the print statement above, the new vocabulary size is 1132 (the
# MAGIC vocabulary size in the previous section was 1130).
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC As an additional quick check, let's print the last 5 entries of the updated vocabulary:
# MAGIC </div>

# COMMAND ----------

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC A simple text tokenizer that handles unknown words</div>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Step 1: Replace unknown words by <|unk|> tokens
# MAGIC     
# MAGIC Step 2: Replace spaces before the specified punctuations
# MAGIC
# MAGIC </div>
# MAGIC

# COMMAND ----------

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]  # token converted to token ids
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# COMMAND ----------

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

# COMMAND ----------

tokenizer.encode(text)


# COMMAND ----------

tokenizer.decode(tokenizer.encode(text))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Based on comparing the de-tokenized text above with the original input text, we know that
# MAGIC the training dataset, Edith Wharton's short story The Verdict, did not contain the words
# MAGIC "Hello" and "palace."
# MAGIC
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC So far, we have discussed tokenization as an essential step in processing text as input to
# MAGIC LLMs. Depending on the LLM, some researchers also consider additional special tokens such
# MAGIC as the following:
# MAGIC
# MAGIC [BOS] (beginning of sequence): This token marks the start of a text. It
# MAGIC signifies to the LLM where a piece of content begins.
# MAGIC
# MAGIC [EOS] (end of sequence): This token is positioned at the end of a text,
# MAGIC and is especially useful when concatenating multiple unrelated texts,
# MAGIC similar to <|endoftext|>. For instance, when combining two different
# MAGIC Wikipedia articles or books, the [EOS] token indicates where one article
# MAGIC ends and the next one begins.
# MAGIC
# MAGIC [PAD] (padding): When training LLMs with batch sizes larger than one,
# MAGIC the batch might contain texts of varying lengths. To ensure all texts have
# MAGIC the same length, the shorter texts are extended or "padded" using the
# MAGIC [PAD] token, up to the length of the longest text in the batch.
# MAGIC
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Note that the tokenizer used for GPT models does not need any of these tokens mentioned
# MAGIC above but only uses an <|endoftext|> token for simplicity
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC the tokenizer used for GPT models also doesn't use an <|unk|> token for outof-vocabulary words. Instead, GPT models use a byte pair encoding tokenizer, which breaks
# MAGIC down words into subword units
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### BYTE PAIR ENCODING
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **BPE Tokenizer**

# COMMAND ----------

! pip3 install tiktoken

# COMMAND ----------

import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

# COMMAND ----------

# MAGIC %md
# MAGIC THE USAGE OF THIS TOKENIZER IS SIMILAR TO SIMPLETOKENIZERV2 WE IMPLEMENTED PREVIOUSLY 
# MAGIC VIA AN ECODER AND DECODER

# COMMAND ----------

tokenizer = tiktoken.get_encoding("gpt2")

# COMMAND ----------

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

# COMMAND ----------

# MAGIC %md
# MAGIC 50256 IS OF ENDOFTEXT, which is token upto 50255

# COMMAND ----------

# MAGIC %md
# MAGIC ID'S TO TOKEN

# COMMAND ----------

strings = tokenizer.decode(integers)

print(strings)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise 2.1**

# COMMAND ----------

integers = tokenizer.encode("BEST ESTIMATE TEST")
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# COMMAND ----------

# MAGIC %md
# MAGIC **Data sampling with sliding window**

# COMMAND ----------

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# COMMAND ----------

enc_sample = enc_text[50:]


# COMMAND ----------

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

# COMMAND ----------

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

# COMMAND ----------

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# COMMAND ----------

# MAGIC %md
# MAGIC **IMPLEMENTING A DATA LOADER**

# COMMAND ----------

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# COMMAND ----------

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# COMMAND ----------

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# COMMAND ----------

import torch
print("PyTorch version:", torch.__version__)
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# COMMAND ----------

second_batch = next(data_iter)
print(second_batch)

# COMMAND ----------

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# COMMAND ----------

# MAGIC %md
# MAGIC **CREATE TOKEN EMBEDDINGS**

# COMMAND ----------

input_ids = torch.tensor([2, 3, 5, 1])


# COMMAND ----------

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# COMMAND ----------

print(embedding_layer.weight)


# COMMAND ----------

print(embedding_layer(torch.tensor([3])))


# COMMAND ----------

print(embedding_layer(input_ids))


# COMMAND ----------

# MAGIC %md
# MAGIC **POSITIONAL EMBEDDINGS (ENCODING WORD POSITIONS)**

# COMMAND ----------

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# COMMAND ----------

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# COMMAND ----------

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# COMMAND ----------

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# COMMAND ----------

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# COMMAND ----------

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# COMMAND ----------

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
