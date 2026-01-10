# Databricks notebook source
# MAGIC %pip install torch

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
# MAGIC

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

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC After determining that the vocabulary size is 1,130 via the above code, we create the
# MAGIC vocabulary and print its first 51 entries for illustration purposes:
# MAGIC
# MAGIC </div>

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

tokenizer = SimpleTokenizerV1(vocab)

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
# MAGIC ### ADDING SPECIAL CONTEXT TOKENS
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

        ids = [self.str_to_int[s] for s in preprocessed]
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
# MAGIC ### BYTE PAIR ENCODING (BPE)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC We implemented a simple tokenization scheme in the previous sections for illustration
# MAGIC purposes. 
# MAGIC
# MAGIC This section covers a more sophisticated tokenization scheme based on a concept
# MAGIC called byte pair encoding (BPE). 
# MAGIC
# MAGIC The BPE tokenizer covered in this section was used to train
# MAGIC LLMs such as GPT-2, GPT-3, and the original model used in ChatGPT.</div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Since implementing BPE can be relatively complicated, we will use an existing Python
# MAGIC open-source library called tiktoken (https://github.com/openai/tiktoken). 
# MAGIC
# MAGIC This library implements
# MAGIC the BPE algorithm very efficiently based on source code in Rust.
# MAGIC </div>

# COMMAND ----------

! pip3 install tiktoken

# COMMAND ----------

import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Once installed, we can instantiate the BPE tokenizer from tiktoken as follows:</div>
# MAGIC

# COMMAND ----------

tokenizer = tiktoken.get_encoding("gpt2")

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC The usage of this tokenizer is similar to SimpleTokenizerV2 we implemented previously via
# MAGIC an encode method:</div>
# MAGIC
# MAGIC

# COMMAND ----------

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC The code above prints the following token IDs:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC We can then convert the token IDs back into text using the decode method, similar to our
# MAGIC SimpleTokenizerV2 earlier:</div>
# MAGIC

# COMMAND ----------

strings = tokenizer.decode(integers)

print(strings)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC We can make two noteworthy observations based on the token IDs and decoded text
# MAGIC above. 
# MAGIC
# MAGIC First, the <|endoftext|> token is assigned a relatively large token ID, namely,
# MAGIC 50256. 
# MAGIC
# MAGIC In fact, the BPE tokenizer, which was used to train models such as GPT-2, GPT-3,
# MAGIC and the original model used in ChatGPT, has a total vocabulary size of 50,257, with
# MAGIC <|endoftext|> being assigned the largest token ID.
# MAGIC     
# MAGIC
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Second, the BPE tokenizer above encodes and decodes unknown words, such as
# MAGIC "someunknownPlace" correctly. 
# MAGIC
# MAGIC The BPE tokenizer can handle any unknown word. How does
# MAGIC it achieve this without using <|unk|> tokens?
# MAGIC     
# MAGIC
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The algorithm underlying BPE breaks down words that aren't in its predefined vocabulary
# MAGIC into smaller subword units or even individual characters.
# MAGIC
# MAGIC The enables it to handle out-ofvocabulary words. 
# MAGIC
# MAGIC So, thanks to the BPE algorithm, if the tokenizer encounters an
# MAGIC unfamiliar word during tokenization, it can represent it as a sequence of subword tokens or
# MAGIC characters
# MAGIC     
# MAGIC
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC **Let us take another simple example to illustrate how the BPE tokenizer deals with unknown tokens**

# COMMAND ----------

integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# COMMAND ----------

import tiktoken

# Initialize the encodings for GPT-2, GPT-3, and GPT-4
encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
    "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
}

# Get the vocabulary size for each encoding
vocab_sizes = {model: encoding.n_vocab for model, encoding in encodings.items()}

# Print the vocabulary sizes
for model, size in vocab_sizes.items():
    print(f"The vocabulary size for {model.upper()} is: {size}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### CREATING INPUT-TARGET PAIRS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC In this section we implement a data loader that fetches the input-target pairs using a sliding window approach.</div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC To get started, we will first tokenize the whole The Verdict short story we worked with
# MAGIC earlier using the BPE tokenizer introduced in the previous section:</div>
# MAGIC
# MAGIC

# COMMAND ----------

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Executing the code above will return 5145, the total number of tokens in the training set,
# MAGIC after applying the BPE tokenizer.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Next, we remove the first 50 tokens from the dataset for demonstration purposes as it
# MAGIC results in a slightly more interesting text passage in the next steps:</div>

# COMMAND ----------

enc_sample = enc_text[50:]


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC One of the easiest and most intuitive ways to create the input-target pairs for the nextword prediction task is to create two variables, x and y, where x contains the input tokens
# MAGIC and y contains the targets, which are the inputs shifted by 1:</div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC The context size determines how many tokens are included in the input
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC

# COMMAND ----------

context_size = 4 #length of the input
#The context_size of 4 means that the model is trained to look at a sequence of 4 words (or tokens) 
#to predict the next word in the sequence. 
#The input x is the first 4 tokens [1, 2, 3, 4], and the target y is the next 4 tokens [2, 3, 4, 5]

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Processing the inputs along with the targets, which are the inputs shifted by one position,
# MAGIC we can then create the next-word prediction tasks as
# MAGIC follows:</div>

# COMMAND ----------

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC Everything left of the arrow (---->) refers to the input an LLM would receive, and the token
# MAGIC ID on the right side of the arrow represents the target token ID that the LLM is supposed to
# MAGIC predict.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC For illustration purposes, let's repeat the previous code but convert the token IDs into
# MAGIC text:</div>

# COMMAND ----------

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC We've now created the input-target pairs that we can turn into use for the LLM training in
# MAGIC upcoming chapters.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC There's only one more task before we can turn the tokens into embeddings:implementing an efficient data loader that
# MAGIC iterates over the input dataset and returns the inputs and targets as PyTorch tensors, which
# MAGIC can be thought of as multidimensional arrays.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In particular, we are interested in returning two tensors: an input tensor containing the
# MAGIC text that the LLM sees and a target tensor that includes the targets for the LLM to predict,
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### IMPLEMENTING A DATA LOADER

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC For the efficient data loader implementation, we will use PyTorch's built-in Dataset and
# MAGIC DataLoader classes.</div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Step 1: Tokenize the entire text
# MAGIC     
# MAGIC Step 2: Use a sliding window to chunk the book into overlapping sequences of max_length
# MAGIC
# MAGIC Step 3: Return the total number of rows in the dataset
# MAGIC
# MAGIC Step 4: Return a single row from the dataset
# MAGIC </div>

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

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The GPTDatasetV1 class in listing 2.5 is based on the PyTorch Dataset class.
# MAGIC
# MAGIC It defines how individual rows are fetched from the dataset. 
# MAGIC
# MAGIC Each row consists of a number of
# MAGIC token IDs (based on a max_length) assigned to an input_chunk tensor. 
# MAGIC
# MAGIC The target_chunk
# MAGIC tensor contains the corresponding targets. 
# MAGIC
# MAGIC I recommend reading on to see how the data
# MAGIC returned from this dataset looks like when we combine the dataset with a PyTorch
# MAGIC DataLoader -- this will bring additional intuition and clarity.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC The following code will use the GPTDatasetV1 to load the inputs in batches via a PyTorch
# MAGIC DataLoader:</div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Step 1: Initialize the tokenizer
# MAGIC
# MAGIC Step 2: Create dataset
# MAGIC
# MAGIC Step 3: drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes
# MAGIC during training
# MAGIC
# MAGIC Step 4: The number of CPU processes to use for preprocessing
# MAGIC     
# MAGIC </div>

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

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC Let's test the dataloader with a batch size of 1 for an LLM with a context size of 4, 
# MAGIC
# MAGIC This will develop an intuition of how the GPTDatasetV1 class and the
# MAGIC create_dataloader_v1 function work together: </div>

# COMMAND ----------

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Convert dataloader into a Python iterator to fetch the next entry via Python's built-in next() function
# MAGIC     
# MAGIC </div>

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

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The first_batch variable contains two tensors: the first tensor stores the input token IDs,
# MAGIC and the second tensor stores the target token IDs. 
# MAGIC
# MAGIC Since the max_length is set to 4, each of the two tensors contains 4 token IDs. 
# MAGIC
# MAGIC Note that an input size of 4 is relatively small and only chosen for illustration purposes. It is common to train LLMs with input sizes of at least
# MAGIC 256.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC To illustrate the meaning of stride=1, let's fetch another batch from this dataset: </div>

# COMMAND ----------

second_batch = next(data_iter)
print(second_batch)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC If we compare the first with the second batch, we can see that the second batch's token
# MAGIC IDs are shifted by one position compared to the first batch. 
# MAGIC
# MAGIC For example, the second ID in
# MAGIC the first batch's input is 367, which is the first ID of the second batch's input. 
# MAGIC
# MAGIC The stride
# MAGIC setting dictates the number of positions the inputs shift across batches, emulating a sliding
# MAGIC window approach
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Batch sizes of 1, such as we have sampled from the data loader so far, are useful for
# MAGIC illustration purposes. 
# MAGIC                                                                                  
# MAGIC If you have previous experience with deep learning, you may know
# MAGIC that small batch sizes require less memory during training but lead to more noisy model
# MAGIC updates.
# MAGIC
# MAGIC Just like in regular deep learning, the batch size is a trade-off and hyperparameter
# MAGIC to experiment with when training LLMs.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC Before we move on to the two final sections of this chapter that are focused on creating
# MAGIC the embedding vectors from the token IDs, let's have a brief look at how we can use the
# MAGIC data loader to sample with a batch size greater than 1: </div>

# COMMAND ----------

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Note that we increase the stride to 4. This is to utilize the data set fully (we don't skip a
# MAGIC single word) but also avoid any overlap between the batches, since more overlap could lead
# MAGIC to increased overfitting.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### CREATING TOKEN EMBEDDINGS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC Let's illustrate how the token ID to embedding vector conversion works with a hands-on
# MAGIC example. Suppose we have the following four input tokens with IDs 2, 3, 5, and 1:</div>

# COMMAND ----------

input_ids = torch.tensor([2, 3, 5, 1])


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC For the sake of simplicity and illustration purposes, suppose we have a small vocabulary of
# MAGIC only 6 words (instead of the 50,257 words in the BPE tokenizer vocabulary), and we want
# MAGIC to create embeddings of size 3 (in GPT-3, the embedding size is 12,288 dimensions):
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC Using the vocab_size and output_dim, we can instantiate an embedding layer in PyTorch,
# MAGIC setting the random seed to 123 for reproducibility purposes:
# MAGIC
# MAGIC </div>

# COMMAND ----------

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC The print statement in the code prints the embedding layer's underlying
# MAGIC weight matrix:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

print(embedding_layer.weight)


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC We can see that the weight matrix of the embedding layer contains small, random values.
# MAGIC These values are optimized during LLM training as part of the LLM optimization itself, as we
# MAGIC will see in upcoming chapters. Moreover, we can see that the weight matrix has six rows
# MAGIC and three columns. There is one row for each of the six possible tokens in the vocabulary.
# MAGIC And there is one column for each of the three embedding dimensions.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC After we instantiated the embedding layer, let's now apply it to a token ID to obtain the
# MAGIC embedding vector:
# MAGIC
# MAGIC </div>

# COMMAND ----------

print(embedding_layer(torch.tensor([3])))


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC If we compare the embedding vector for token ID 3 to the previous embedding matrix, we
# MAGIC see that it is identical to the 4th row (Python starts with a zero index, so it's the row
# MAGIC corresponding to index 3). In other words, the embedding layer is essentially a look-up
# MAGIC operation that retrieves rows from the embedding layer's weight matrix via a token ID.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC Previously, we have seen how to convert a single token ID into a three-dimensional
# MAGIC embedding vector. Let's now apply that to all four input IDs we defined earlier
# MAGIC (torch.tensor([2, 3, 5, 1])):
# MAGIC
# MAGIC </div>

# COMMAND ----------

print(embedding_layer(input_ids))


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Each row in this output matrix is obtained via a lookup operation from the embedding
# MAGIC weight matrix
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### POSITIONAL EMBEDDINGS (ENCODING WORD POSITIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Previously, we focused on very small embedding sizes in this chapter for illustration
# MAGIC purposes. 
# MAGIC
# MAGIC We now consider more realistic and useful embedding sizes and encode the input
# MAGIC tokens into a 256-dimensional vector representation. 
# MAGIC
# MAGIC This is smaller than what the original
# MAGIC GPT-3 model used (in GPT-3, the embedding size is 12,288 dimensions) but still reasonable
# MAGIC for experimentation. 
# MAGIC
# MAGIC Furthermore, we assume that the token IDs were created by the BPE
# MAGIC tokenizer that we implemented earlier, which has a vocabulary size of 50,257:
# MAGIC
# MAGIC </div>

# COMMAND ----------

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Using the token_embedding_layer above, if we sample data from the data loader, we
# MAGIC embed each token in each batch into a 256-dimensional vector. If we have a batch size of 8
# MAGIC with four tokens each, the result will be an 8 x 4 x 256 tensor.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's instantiate the data loader ( Data sampling with a sliding window),
# MAGIC first:
# MAGIC
# MAGIC </div>

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

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC As we can see, the token ID tensor is 8x4-dimensional, meaning that the data batch
# MAGIC consists of 8 text samples with 4 tokens each.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's now use the embedding layer to embed these token IDs into 256-dimensional
# MAGIC vectors:
# MAGIC
# MAGIC </div>

# COMMAND ----------

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC As we can tell based on the 8x4x256-dimensional tensor output, each token ID is now
# MAGIC embedded as a 256-dimensional vector.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC For a GPT model's absolute embedding approach, we just need to create another
# MAGIC embedding layer that has the same dimension as the token_embedding_layer:
# MAGIC
# MAGIC </div>

# COMMAND ----------

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# COMMAND ----------

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC As shown in the preceding code example, the input to the pos_embeddings is usually a
# MAGIC placeholder vector torch.arange(context_length), which contains a sequence of
# MAGIC numbers 0, 1, ..., up to the maximum input length − 1. 
# MAGIC
# MAGIC The context_length is a variable
# MAGIC that represents the supported input size of the LLM. 
# MAGIC
# MAGIC Here, we choose it similar to the
# MAGIC maximum length of the input text. 
# MAGIC
# MAGIC In practice, input text can be longer than the supported
# MAGIC context length, in which case we have to truncate the text.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC As we can see, the positional embedding tensor consists of four 256-dimensional vectors.
# MAGIC We can now add these directly to the token embeddings, where PyTorch will add the 4x256-
# MAGIC dimensional pos_embeddings tensor to each 4x256-dimensional token embedding tensor in
# MAGIC each of the 8 batches:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The input_embeddings we created are the embedded input
# MAGIC examples that can now be processed by the main LLM modules
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## IMPLEMENTING A SIMPLIFIED ATTENTION MECHANISM

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Consider the following input sentence, which has already been embedded into 3-
# MAGIC dimensional vectors. 
# MAGIC
# MAGIC We choose a small embedding dimension for
# MAGIC illustration purposes to ensure it fits on the page without line breaks:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %pip install torch

# COMMAND ----------

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# COMMAND ----------

# Create 3D plot with vectors from origin to each point, using different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a list of colors for the vectors
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Plot each vector with a different color and annotate with the corresponding word
for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
    # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits to keep arrows within the plot boundaries
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Each row represents a word, and each column represents an embedding dimension
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC The second input token serves as the query    
# MAGIC </div>

# COMMAND ----------

query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC In the next step, we normalize each of the attention scores that
# MAGIC we computed previously.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The main goal behind the normalization  is to obtain attention weights
# MAGIC that sum up to 1. 
# MAGIC
# MAGIC This normalization is a convention that is useful for interpretation and for
# MAGIC maintaining training stability in an LLM. 
# MAGIC
# MAGIC Here's a straightforward method for achieving this
# MAGIC normalization step:
# MAGIC
# MAGIC </div>

# COMMAND ----------

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC In practice, it's more common and advisable to use the softmax function for normalization.
# MAGIC
# MAGIC This approach is better at managing extreme values and offers more favorable gradient
# MAGIC properties during training. 
# MAGIC
# MAGIC Below is a basic implementation of the softmax function for
# MAGIC normalizing the attention scores: 
# MAGIC </div>

# COMMAND ----------

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC As the output shows, the softmax function also meets the objective and normalizes the
# MAGIC attention weights such that they sum to 1:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In addition, the softmax function ensures that the attention weights are always positive.
# MAGIC This makes the output interpretable as probabilities or relative importance, where higher
# MAGIC weights indicate greater importance.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Note that this naive softmax implementation (softmax_naive) may encounter numerical
# MAGIC instability problems, such as overflow and underflow, when dealing with large or small input
# MAGIC values. 
# MAGIC
# MAGIC Therefore, in practice, it's advisable to use the PyTorch implementation of softmax,
# MAGIC which has been extensively optimized for performance:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC In this case, we can see that it yields the same results as our previous softmax_naive
# MAGIC function:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The context vector z(2)is calculated as a weighted sum of all input
# MAGIC vectors. 
# MAGIC
# MAGIC This involves multiplying each input vector by its corresponding attention weight:
# MAGIC
# MAGIC </div>

# COMMAND ----------

query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

# COMMAND ----------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputs2 = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55], # step     (x^6)
   [0.4419, 0.6515, 0.5683]]
)

# Corresponding words
words2 = ['Your', 'journey', 'starts', 'with', 'one', 'step', 'journey-context']

# Extract x, y, z coordinates
x_coords = inputs2[:, 0].numpy()
y_coords = inputs2[:, 1].numpy()
z_coords = inputs2[:, 2].numpy()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point and annotate with corresponding word
for x, y, z, word in zip(x_coords, y_coords, z_coords, words2):
    ax.scatter(x, y, z)
    ax.text(x, y, z, word, fontsize=10)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('3D Plot of Word Embeddings')
plt.show()

# Create 3D plot with vectors from origin to each point, using different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a list of colors for the vectors
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'r']

# Plot each vector with a different color and annotate with the corresponding word
for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words2, colors):
    # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits to keep arrows within the plot boundaries
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Now, we can extend this computation to
# MAGIC calculate attention weights and context vectors for all inputs.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC First, we add an additional for-loop to compute the
# MAGIC dot products for all pairs of inputs.
# MAGIC
# MAGIC </div>

# COMMAND ----------

attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Each element in the preceding tensor represents an attention score between each pair of
# MAGIC inputs.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC When computing the preceding attention score tensor, we used for-loops in Python.
# MAGIC                                                             
# MAGIC However, for-loops are generally slow, and we can achieve the same results using matrix
# MAGIC multiplication:
# MAGIC </div>

# COMMAND ----------

attn_scores = inputs @ inputs.T
print(attn_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC We now normalize each row so that the values in
# MAGIC each row sum to 1:
# MAGIC
# MAGIC </div>

# COMMAND ----------

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In the context of using PyTorch, the dim parameter in functions like torch.softmax specifies
# MAGIC the dimension of the input tensor along which the function will be computed. 
# MAGIC
# MAGIC By setting
# MAGIC dim=-1, we are instructing the softmax function to apply the normalization along the last
# MAGIC dimension of the attn_scores tensor. 
# MAGIC
# MAGIC If attn_scores is a 2D tensor (for example, with a
# MAGIC shape of [rows, columns]), dim=-1 will normalize across the columns so that the values in
# MAGIC each row (summing over the column dimension) sum up to 1.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's briefly verify that
# MAGIC the rows indeed all sum to 1:
# MAGIC
# MAGIC </div>

# COMMAND ----------

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC In the third and last step, we now use these attention weights to compute all context
# MAGIC vectors via matrix multiplication:
# MAGIC
# MAGIC </div>

# COMMAND ----------

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC We can double-check that the code is correct by comparing the 2nd row with the context
# MAGIC vector z(2) calculated previously
# MAGIC
# MAGIC </div>

# COMMAND ----------

print("Previous 2nd context vector:", context_vec_2)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Based on the result, we can see that the previously calculated context_vec_2 matches the
# MAGIC second row in the previous tensor exactly
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC This concludes the code walkthrough of a simple self-attention mechanism.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## IMPLEMENTING SELF ATTENTION WITH TRAINABLE WEIGHTS

# COMMAND ----------

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's begin by defining a few variables:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC #A The second input element
# MAGIC
# MAGIC #B The input embedding size, d=3
# MAGIC
# MAGIC
# MAGIC #C The output embedding size, d_out=2
# MAGIC
# MAGIC </div>

# COMMAND ----------

x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Note that in GPT-like models, the input and output dimensions are usually the same. 
# MAGIC
# MAGIC But for illustration purposes, to better follow the computation, we choose different input (d_in=3)
# MAGIC and output (d_out=2) dimensions here.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Next, we initialize the three weight matrices Wq, Wk and Wv
# MAGIC
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# COMMAND ----------

print(W_query)

# COMMAND ----------

print(W_key)

# COMMAND ----------

print(W_value)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC Note that we are setting requires_grad=False to reduce clutter in the outputs for
# MAGIC illustration purposes. 
# MAGIC
# MAGIC If we were to use the weight matrices for model training, we
# MAGIC would set requires_grad=True to update these matrices during model training.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Next, we compute the query, key, and value vectors as shown earlier
# MAGIC </div>

# COMMAND ----------

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC As we can see based on the output for the query, this results in a 2-dimensional vector. 
# MAGIC
# MAGIC This is because: we set the number of columns of the corresponding weight matrix, via d_out, to 2:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Even though our temporary goal is to only compute the one context vector z(2),  we still
# MAGIC require the key and value vectors for all input elements. 
# MAGIC
# MAGIC This is because they are involved in computing the attention weights with respect to the query q(2)
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC We can obtain all keys and values via matrix multiplication:
# MAGIC </div>

# COMMAND ----------

keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query
print("keys.shape:", keys.shape)

print("values.shape:", values.shape)

print("queries.shape:", queries.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC     
# MAGIC As we can tell from the outputs, we successfully projected the 6 input tokens from a 3D
# MAGIC onto a 2D embedding space:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC First, let's compute the attention score ω22</div>

# COMMAND ----------

keys_2 = keys[1] #A
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Again, we can generalize this computation to all attention scores via matrix multiplication:</div>

# COMMAND ----------

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)

# COMMAND ----------

attn_scores = queries @ keys.T # omega
print(attn_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC We compute the attention weights by scaling the
# MAGIC attention scores and using the softmax function we used earlier. 
# MAGIC
# MAGIC The difference to earlier is
# MAGIC that we now scale the attention scores by dividing them by the square root of the
# MAGIC embedding dimension of the keys. 
# MAGIC
# MAGIC Note that taking the square root is mathematically the
# MAGIC same as exponentiating by 0.5:</div>

# COMMAND ----------

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
print(d_k)

# COMMAND ----------

# MAGIC %md
# MAGIC ## WHY DIVIDE BY SQRT (DIMENSION)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Reason 1: For stability in learning
# MAGIC
# MAGIC The softmax function is sensitive to the magnitudes of its inputs. When the inputs are large, the differences between the exponential values of each input become much more pronounced. This causes the softmax output to become "peaky," where the highest value receives almost all the probability mass, and the rest receive very little.
# MAGIC
# MAGIC In attention mechanisms, particularly in transformers, if the dot products between query and key vectors become too large (like multiplying by 8 in this example), the attention scores can become very large. This results in a very sharp softmax distribution, making the model overly confident in one particular "key." Such sharp distributions can make learning unstable,
# MAGIC     
# MAGIC </div>

# COMMAND ----------

import torch

# Define the tensor
tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

# Apply softmax without scaling
softmax_result = torch.softmax(tensor, dim=-1)
print("Softmax without scaling:", softmax_result)

# Multiply the tensor by 8 and then apply softmax
scaled_tensor = tensor * 8
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
print("Softmax after scaling (tensor * 8):", softmax_scaled_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## BUT WHY SQRT?

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Reason 2: To make the variance of the dot product stable
# MAGIC
# MAGIC The dot product of  Q and K increases the variance because multiplying two random numbers increases the variance.
# MAGIC
# MAGIC The increase in variance grows with the dimension. 
# MAGIC
# MAGIC Dividing by sqrt (dimension) keeps the variance close to 1
# MAGIC     
# MAGIC </div>

# COMMAND ----------

import numpy as np

# Function to compute variance before and after scaling
def compute_variance(dim, num_trials=1000):
    dot_products = []
    scaled_dot_products = []

    # Generate multiple random vectors and compute dot products
    for _ in range(num_trials):
        q = np.random.randn(dim)
        k = np.random.randn(dim)
        
        # Compute dot product
        dot_product = np.dot(q, k)
        dot_products.append(dot_product)
        
        # Scale the dot product by sqrt(dim)
        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)
    
    # Calculate variance of the dot products
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling

# For dimension 5
variance_before_5, variance_after_5 = compute_variance(5)
print(f"Variance before scaling (dim=5): {variance_before_5}")
print(f"Variance after scaling (dim=5): {variance_after_5}")

# For dimension 20
variance_before_100, variance_after_100 = compute_variance(100)
print(f"Variance before scaling (dim=100): {variance_before_100}")
print(f"Variance after scaling (dim=100): {variance_after_100}")



# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC We now compute the context vector as a weighted sum over the value
# MAGIC vectors. 
# MAGIC
# MAGIC Here, the attention weights serve as a weighting factor that weighs the respective
# MAGIC importance of each value vector. 
# MAGIC
# MAGIC We can use matrix multiplication to
# MAGIC obtain the output in one step:</div>

# COMMAND ----------

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC So far, we only computed a single context vector, z(2). 
# MAGIC
# MAGIC In the next section, we will generalize the code to compute all context vectors in the input sequence, z(1)to z (T)</div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## IMPLEMENTING A COMPACT SELF ATTENTION PYTHON CLASS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC In the previous sections, we have gone through a lot of steps to compute the self-attention
# MAGIC outputs. 
# MAGIC
# MAGIC This was mainly done for illustration purposes so we could go through one step at
# MAGIC a time. 
# MAGIC
# MAGIC In practice, with the LLM implementation in the next chapter in mind, it is helpful to
# MAGIC organize this code into a Python class as follows:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In this PyTorch code, SelfAttention_v1 is a class derived from nn.Module, which is a
# MAGIC fundamental building block of PyTorch models, which provides necessary functionalities for
# MAGIC model layer creation and management.    
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The __init__ method initializes trainable weight matrices (W_query, W_key, and
# MAGIC W_value) for queries, keys, and values, each transforming the input dimension d_in to an
# MAGIC output dimension d_out.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC During the forward pass, using the forward method, we compute the attention scores
# MAGIC (attn_scores) by multiplying queries and keys, normalizing these scores using softmax.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     
# MAGIC Finally, we create a context vector by weighting the values with these normalized attention
# MAGIC scores.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Since inputs contains six embedding vectors, we get a matrix storing the six
# MAGIC context vectors, as shown in the above result. 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As a quick check, notice how the second row ([0.3061, 0.8210]) matches the contents of
# MAGIC context_vec_2 in the previous section.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC We can improve the SelfAttention_v1 implementation further by utilizing PyTorch's
# MAGIC nn.Linear layers, which effectively perform matrix multiplication when the bias units are
# MAGIC disabled. 
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Additionally, a significant advantage of using nn.Linear instead of manually
# MAGIC implementing nn.Parameter(torch.rand(...)) is that nn.Linear has an optimized weight
# MAGIC initialization scheme, contributing to more stable and effective model training.
# MAGIC
# MAGIC </div>

# COMMAND ----------

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC You can use the SelfAttention_v2 similar to SelfAttention_v1:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Note that SelfAttention_v1 and SelfAttention_v2 give different outputs because they
# MAGIC use different initial weights for the weight matrices since nn.Linear uses a more
# MAGIC sophisticated weight initialization scheme.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## HIDING FUTURE WORDS WITH CAUSAL ATTENTION

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's work with the attention scores and weights from the previous section to code the causal attention mechanism.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC In the first step illustrated in Figure 3.20, we compute the attention weights using the
# MAGIC softmax function as we have done in previous sections:    
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Reuse the query and key weight matrices of the SelfAttention_v2 object from the previous section for
# MAGIC convenience
# MAGIC     
# MAGIC </div>

# COMMAND ----------

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# COMMAND ----------

queries = sa_v2.W_query(inputs) #A
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC We can now use PyTorch's tril function to create a mask
# MAGIC where the values above the diagonal are zero:
# MAGIC
# MAGIC </div>

# COMMAND ----------

torch.ones(context_length, context_length)

# COMMAND ----------

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Now, we can multiply this mask with the attention weights to zero out the values above the
# MAGIC diagonal:
# MAGIC
# MAGIC </div>

# COMMAND ----------

masked_simple = attn_weights*mask_simple
print(masked_simple)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see, the elements above the diagonal are successfully zeroed out
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The third step is to renormalize the attention weights to sum up to 1 again in
# MAGIC each row. 
# MAGIC
# MAGIC We can achieve this by dividing each element in each row by the sum in each
# MAGIC row:
# MAGIC
# MAGIC </div>

# COMMAND ----------

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC The result is an attention weight matrix where the attention weights above the diagonal are
# MAGIC zeroed out and where the rows sum to 1.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC While we could be technically done with implementing causal attention at this point, we can
# MAGIC take advantage of a mathematical property of the softmax function. 
# MAGIC
# MAGIC We can implement the computation of the masked attention weights more efficiently in fewer steps.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The softmax function converts its inputs into a probability distribution. 
# MAGIC
# MAGIC When negative
# MAGIC infinity values (-∞) are present in a row, the softmax function treats them as zero
# MAGIC probability. 
# MAGIC
# MAGIC (Mathematically, this is because e
# MAGIC -∞ approaches 0.)
# MAGIC
# MAGIC
# MAGIC We can implement this more efficient masking "trick" by creating a mask with 1's above
# MAGIC the diagonal and then replacing these 1's with negative infinity (-inf) values:
# MAGIC
# MAGIC </div>

# COMMAND ----------

print(attn_scores)

# COMMAND ----------

torch.triu(torch.ones(context_length, context_length))

# COMMAND ----------

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)

# COMMAND ----------

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Now, all we need to do is apply the softmax function to these masked results, and we are
# MAGIC done.
# MAGIC
# MAGIC </div>

# COMMAND ----------

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see based on the output, the values in each row sum to 1, and no further
# MAGIC normalization is necessary.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Masking in Transformers sets scores for future tokens to a large negative value, making their influence in the softmax calculation effectively zero. 
# MAGIC
# MAGIC The softmax function then recalculates attention weights only among the unmasked tokens. 
# MAGIC
# MAGIC This process ensures no information leakage from masked tokens, focusing the model solely on the intended data.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC We could now use the modified attention weights to compute the context vectors via
# MAGIC context_vec = attn_weights @ values.
# MAGIC
# MAGIC However, in the next section,
# MAGIC we first cover another minor tweak to the causal attention mechanism that is useful for
# MAGIC reducing overfitting when training LLMs.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### MASKING ADDITIONAL ATTENTION WEIGHTS WITH DROPOUT

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC In the following code example, we use a dropout rate of 50%, which means masking out
# MAGIC half of the attention weights.
# MAGIC
# MAGIC When we train the GPT model in later chapters, we will use a
# MAGIC lower dropout rate, such as 0.1 or 0.2.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC In the following code, we apply PyTorch's dropout implementation first to a 6×6 tensor
# MAGIC consisting of ones for illustration purposes:
# MAGIC </div>

# COMMAND ----------

example = torch.ones(6, 6) #B
print(example)

# COMMAND ----------

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #A
example = torch.ones(6, 6) #B
print(dropout(example))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC When applying dropout to an attention weight matrix with a rate of 50%, half of the
# MAGIC elements in the matrix are randomly set to zero. 
# MAGIC
# MAGIC To compensate for the reduction in active
# MAGIC elements, the values of the remaining elements in the matrix are scaled up by a factor of
# MAGIC 1/0.5 =2. 
# MAGIC
# MAGIC This scaling is crucial to maintain the overall balance of the attention weights,
# MAGIC ensuring that the average influence of the attention mechanism remains consistent during
# MAGIC both the training and inference phases.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Now, let's apply dropout to the attention weight matrix itself:
# MAGIC
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(123)
print(dropout(attn_weights))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see above, the resulting attention weight matrix now has additional elements zeroed out and the
# MAGIC remaining ones rescaled.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Having gained an understanding of causal attention and dropout masking, we will
# MAGIC develop a concise Python class in the following section. 
# MAGIC
# MAGIC This class is designed to facilitate
# MAGIC the efficient application of these two techniques.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### IMPLEMENTING A COMPACT CAUSAL ATTENTION CLASS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC In this section, we will now incorporate the causal attention and dropout modifications into
# MAGIC the SelfAttention Python class we developed in section 3.4. 
# MAGIC
# MAGIC This class will then serve as a
# MAGIC template for developing multi-head attention in the upcoming section.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Before we begin, one more thing is to ensure that the code can handle batches
# MAGIC consisting of more than one input. 
# MAGIC
# MAGIC This will ensure that the CausalAttention class supports the batch
# MAGIC outputs produced by the data loader we implemented earlier.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC For simplicity, to simulate such batch inputs, we duplicate the input text example:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC  2 inputs with 6 tokens each, and each token has embedding dimension 3
# MAGIC     
# MAGIC </div>

# COMMAND ----------

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC This results in a 3D tensor consisting of 2 input texts with 6 tokens each, where each token
# MAGIC is a 3-dimensional embedding vector.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The following CausalAttention class is similar to the SelfAttention class we
# MAGIC implemented earlier, except that we now added the dropout and causal mask components
# MAGIC as highlighted in the following code.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Step 1: Compared to the previous SelfAttention_v1 class, we added a dropout layer.
# MAGIC     
# MAGIC Step 2: The register_buffer call is also a new addition (more information is provided in the following text).
# MAGIC
# MAGIC Step 3:  We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
# MAGIC
# MAGIC Step 4: In PyTorch, operations with a trailing underscore are performed in-place, avoiding unnecessary memory
# MAGIC copies
# MAGIC     
# MAGIC </div>

# COMMAND ----------

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The use of register_buffer in
# MAGIC PyTorch is not strictly necessary for all use cases but offers several advantages here. 
# MAGIC
# MAGIC For
# MAGIC instance, when we use the CausalAttention class in our LLM, buffers are automatically
# MAGIC moved to the appropriate device (CPU or GPU) along with our model, which will be relevant
# MAGIC when training the LLM in future chapters. 
# MAGIC
# MAGIC This means we don't need to manually ensure
# MAGIC these tensors are on the same device as your model parameters, avoiding device mismatch
# MAGIC errors.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC We can use the CausalAttention class as follows, similar to SelfAttention previously:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

print(d_in)

# COMMAND ----------

print(d_out)

# COMMAND ----------

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)

# COMMAND ----------

print(context_vecs)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see, the resulting context vector is a 3D tensor where each token is now represented by a 2D
# MAGIC embedding:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In the next section, we will expand on this concept
# MAGIC and implement a multi-head attention module, that implements several of such causal
# MAGIC attention mechanisms in parallel.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXTENDING SINGLE HEAD ATTENTION TO MULTI-HEAD ATTENTION

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC In practical terms, implementing multi-head attention involves creating multiple instances
# MAGIC of the self-attention mechanism, each with
# MAGIC its own weights, and then combining their outputs
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC In code, we can achieve this by implementing a simple MultiHeadAttentionWrapper
# MAGIC class that stacks multiple instances of our previously implemented CausalAttention
# MAGIC module:
# MAGIC     
# MAGIC </div>

# COMMAND ----------

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC For example, if we use this MultiHeadAttentionWrapper class with two attention heads (via
# MAGIC num_heads=2) and CausalAttention output dimension d_out=2, this results in a 4-
# MAGIC dimensional context vectors (d_out*num_heads=4)
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC To illustrate further with a concrete example, we can use the
# MAGIC MultiHeadAttentionWrapper class similar to the CausalAttention class before:
# MAGIC </div>

# COMMAND ----------

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 

# COMMAND ----------

torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens = 6
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC The first dimension of the resulting context_vecs tensor is 2 since we have two input texts
# MAGIC (the input texts are duplicated, which is why the context vectors are exactly the same for
# MAGIC those). 
# MAGIC
# MAGIC The second dimension refers to the 6 tokens in each input. The third dimension
# MAGIC refers to the 4-dimensional embedding of each token.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC In this section, we implemented a MultiHeadAttentionWrapper that combined multiple
# MAGIC single-head attention modules. 
# MAGIC
# MAGIC However, note that these are processed sequentially via
# MAGIC [head(x) for head in self.heads] in the forward method. 
# MAGIC
# MAGIC We can improve this
# MAGIC implementation by processing the heads in parallel. 
# MAGIC
# MAGIC One way to achieve this is by
# MAGIC computing the outputs for all attention heads simultaneously via matrix multiplication, as
# MAGIC we will explore in the next section.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### IMPLEMENTING MULTI-HEAD ATTENTION WITH WEIGHT SPLITS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC Instead of maintaining two separate classes, MultiHeadAttentionWrapper and
# MAGIC CausalAttention, we can combine both of these concepts into a single
# MAGIC MultiHeadAttention class. 
# MAGIC
# MAGIC Also, in addition to just merging the
# MAGIC MultiHeadAttentionWrapper with the CausalAttention code, we will make some other
# MAGIC modifications to implement multi-head attention more efficiently.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC In the MultiHeadAttentionWrapper, multiple heads are implemented by creating a list
# MAGIC of CausalAttention objects (self.heads), each representing a separate attention head.
# MAGIC
# MAGIC
# MAGIC The CausalAttention class independently performs the attention mechanism, and the
# MAGIC results from each head are concatenated.
# MAGIC
# MAGIC In contrast, the following MultiHeadAttention
# MAGIC class integrates the multi-head functionality within a single class. 
# MAGIC
# MAGIC
# MAGIC It splits the input into
# MAGIC multiple heads by reshaping the projected query, key, and value tensors and then combines
# MAGIC the results from these heads after computing attention.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's take a look at the MultiHeadAttention class before we discuss it further:
# MAGIC </div>

# COMMAND ----------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Step 1: Reduce the projection dim to match desired output dim
# MAGIC
# MAGIC Step 2: Use a Linear layer to combine head outputs
# MAGIC
# MAGIC Step 3: Tensor shape: (b, num_tokens, d_out)
# MAGIC
# MAGIC Step 4: We implicitly split the matrix by adding a `num_heads` dimension. Then we unroll last dim: (b,
# MAGIC num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
# MAGIC
# MAGIC Step 5: Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
# MAGIC
# MAGIC Step 6: Compute dot product for each head
# MAGIC
# MAGIC Step 7: Mask truncated to the number of tokens
# MAGIC
# MAGIC Step 8: Use the mask to fill attention scores
# MAGIC
# MAGIC Step 9: Tensor shape: (b, num_tokens, n_heads, head_dim)
# MAGIC
# MAGIC Step 10: Combine heads, where self.d_out = self.num_heads * self.head_dim
# MAGIC
# MAGIC Step 11: Add an optional linear projection
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Even though the reshaping (.view) and transposing (.transpose) of tensors inside the
# MAGIC MultiHeadAttention class looks very complicated, mathematically, the
# MAGIC MultiHeadAttention class implements the same concept as the
# MAGIC MultiHeadAttentionWrapper earlier.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC On a big-picture level, in the previous MultiHeadAttentionWrapper, we stacked
# MAGIC multiple single-head attention layers that we combined into a multi-head attention layer.
# MAGIC
# MAGIC
# MAGIC The MultiHeadAttention class takes an integrated approach. 
# MAGIC
# MAGIC It starts with a multi-head
# MAGIC layer and then internally splits this layer into individual attention heads
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC #### DETAILED EXPLANATION OF THE MULTI-HEAD ATTENTION CLASS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The splitting of the query, key, and value tensors, is achieved
# MAGIC through tensor reshaping and transposing operations using PyTorch's .view and
# MAGIC .transpose methods. 
# MAGIC
# MAGIC The input is first transformed (via linear layers for queries, keys, and
# MAGIC values) and then reshaped to represent multiple heads.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The key operation is to split the d_out dimension into num_heads and head_dim, where
# MAGIC head_dim = d_out / num_heads. 
# MAGIC
# MAGIC This splitting is then achieved using the .view method: a
# MAGIC tensor of dimensions (b, num_tokens, d_out) is reshaped to dimension (b, num_tokens,
# MAGIC num_heads, head_dim).
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The tensors are then transposed to bring the num_heads dimension before the
# MAGIC num_tokens dimension, resulting in a shape of (b, num_heads, num_tokens, head_dim).
# MAGIC
# MAGIC This transposition is crucial for correctly aligning the queries, keys, and values across the
# MAGIC different heads and performing batched matrix multiplications efficiently.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC To illustrate this batched matrix multiplication, suppose we have the following example
# MAGIC tensor:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Continuing with MultiHeadAttention, after computing the attention weights and context
# MAGIC vectors, the context vectors from all heads are transposed back to the shape (b,
# MAGIC num_tokens, num_heads, head_dim). 
# MAGIC
# MAGIC These vectors are then reshaped (flattened) into the
# MAGIC shape (b, num_tokens, d_out), effectively combining the outputs from all heads
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Additionally, we added a so-called output projection layer (self.out_proj) to
# MAGIC MultiHeadAttention after combining the heads, which is not present in the
# MAGIC CausalAttention class. 
# MAGIC
# MAGIC This output projection layer is not strictly necessary, but it is commonly used in many LLM
# MAGIC architectures, which is why we added it here for completeness.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Even though the MultiHeadAttention class looks more complicated than the
# MAGIC MultiHeadAttentionWrapper due to the additional reshaping and transposition of tensors,
# MAGIC it is more efficient. 
# MAGIC
# MAGIC The reason is that we only need one matrix multiplication to compute
# MAGIC the keys, for instance, keys = self.W_key(x) (the same is true for the queries and
# MAGIC values). 
# MAGIC                                               
# MAGIC
# MAGIC In the MultiHeadAttentionWrapper, we needed to repeat this matrix multiplication,
# MAGIC which is computationally one of the most expensive steps, for each attention head.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC The MultiHeadAttention class can be used similar to the SelfAttention and
# MAGIC CausalAttention classes we implemented earlier:
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(123)

# Define the tensor with 3 rows and 6 columns
inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 

batch_size, context_length, d_in = batch.shape
d_out = 6
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see based on the results, the output dimension is directly controlled by the
# MAGIC d_out argument:
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In this section, we implemented the MultiHeadAttention class that we will use in the
# MAGIC upcoming sections when implementing and training the LLM itself. 
# MAGIC
# MAGIC
# MAGIC Note that while the code is fully functional, we used relatively small embedding sizes and numbers of attention
# MAGIC heads to keep the outputs readable.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC For comparison, the smallest GPT-2 model (117 million parameters) has 12 attention
# MAGIC heads and a context vector embedding size of 768. 
# MAGIC
# MAGIC The largest GPT-2 model (1.5 billion
# MAGIC parameters) has 25 attention heads and a context vector embedding size of 1600.
# MAGIC
# MAGIC Note
# MAGIC that the embedding sizes of the token inputs and context embeddings are the same in GPT
# MAGIC models (d_in = d_out).
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## IMPLEMENTING A GPT MODEL FROM SCRATCH TO GENERATE TEXT

# COMMAND ----------

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT ARCHITECTURE PART 1: DUMMY GPT MODEL CLASS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Step 1: Use a placeholder for TransformerBlock
# MAGIC
# MAGIC Step 2: Use a placeholder for LayerNorm
# MAGIC </div>

# COMMAND ----------

# MAGIC %pip install torch

# COMMAND ----------

import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape #
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The DummyGPTModel class in this code defines a simplified version of a GPT-like model using
# MAGIC PyTorch's neural network module (nn.Module). 
# MAGIC
# MAGIC The model architecture in the
# MAGIC DummyGPTModel class consists of token and positional embeddings, dropout, a series of
# MAGIC transformer blocks (DummyTransformerBlock), a final layer normalization
# MAGIC (DummyLayerNorm), and a linear output layer (out_head). 
# MAGIC
# MAGIC The configuration is passed in via
# MAGIC a Python dictionary, for instance, the GPT_CONFIG_124M dictionary we created earlier.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC The forward method describes the data flow through the model: it computes token and
# MAGIC positional embeddings for the input indices, applies dropout, processes the data through
# MAGIC the transformer blocks, applies normalization, and finally produces logits with the linear
# MAGIC output layer.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The code above is already functional, as we will see later in this section after we prepare
# MAGIC the input data. 
# MAGIC
# MAGIC However, for now, note in the code above that we have used placeholders
# MAGIC (DummyLayerNorm and DummyTransformerBlock) for the transformer block and layer
# MAGIC normalization, which we will develop in later sections
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Next, we will prepare the input data and initialize a new GPT model to illustrate its
# MAGIC usage.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 1: TOKENIZATION

# COMMAND ----------

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# COMMAND ----------

# MAGIC %md
# MAGIC ### STEP 2: CREATE AN INSTANCE OF DUMMYGPTMODEL

# COMMAND ----------

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The output tensor has two rows corresponding to the two text samples. Each text sample
# MAGIC consists of 4 tokens; each token is a 50,257-dimensional vector, which matches the size of
# MAGIC the tokenizer's vocabulary.
# MAGIC
# MAGIC
# MAGIC The embedding has 50,257 dimensions because each of these dimensions refers to a
# MAGIC unique token in the vocabulary. At the end of this chapter, when we implement the
# MAGIC postprocessing code, we will convert these 50,257-dimensional vectors back into token IDs,
# MAGIC which we can then decode into words.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Now that we have taken a top-down look at the GPT architecture and its in- and outputs,
# MAGIC we will code the individual placeholders in the upcoming sections, starting with the real
# MAGIC layer normalization class that will replace the DummyLayerNorm in the previous code.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT ARCHITECTURE PART 2: LAYER NORMALIZATION

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explanation with a simple example

# COMMAND ----------

torch.manual_seed(123)
batch_example = torch.randn(2, 5) #A
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC The neural network layer we have coded consists of a Linear layer followed by a non-linear
# MAGIC activation function, ReLU (short for Rectified Linear Unit), which is a standard activation
# MAGIC function in neural networks. 
# MAGIC
# MAGIC If you are unfamiliar with ReLU, it simply thresholds negative
# MAGIC inputs to 0, ensuring that a layer outputs only positive values, which explains why the
# MAGIC resulting layer output does not contain any negative values. 
# MAGIC
# MAGIC (Note that we will use another,
# MAGIC more sophisticated activation function in GPT, which we will introduce in the next section).
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Before we apply layer normalization to these outputs, let's examine the mean and
# MAGIC variance:
# MAGIC
# MAGIC </div>

# COMMAND ----------

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The first row in the mean tensor above contains the mean value for the first input row, and
# MAGIC the second output row contains the mean for the second input row.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Using keepdim=True in operations like mean or variance calculation ensures that the
# MAGIC output tensor retains the same number of dimensions as the input tensor, even though the
# MAGIC operation reduces the tensor along the dimension specified via dim. 
# MAGIC
# MAGIC For instance, without
# MAGIC keepdim=True, the returned mean tensor would be a 2-dimensional vector [0.1324,
# MAGIC 0.2170] instead of a 2×1-dimensional matrix [[0.1324], [0.2170]].
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC For a 2D tensor (like a matrix), using dim=-1 for operations such as
# MAGIC mean or variance calculation is the same as using dim=1. 
# MAGIC
# MAGIC This is because -1 refers to the
# MAGIC tensor's last dimension, which corresponds to the columns in a 2D tensor. 
# MAGIC
# MAGIC Later, when
# MAGIC adding layer normalization to the GPT model, which produces 3D tensors with shape
# MAGIC [batch_size, num_tokens, embedding_size], we can still use dim=-1 for normalization
# MAGIC across the last dimension, avoiding a change from dim=1 to dim=2.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Next, let us apply layer normalization to the layer outputs we obtained earlier. The
# MAGIC operation consists of subtracting the mean and dividing by the square root of the variance
# MAGIC (also known as standard deviation):
# MAGIC
# MAGIC </div>

# COMMAND ----------

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Note that the value 2.9802e-08 in the output tensor is the scientific notation for 2.9802 ×
# MAGIC 10-8, which is 0.0000000298 in decimal form. This value is very close to 0, but it is not
# MAGIC exactly 0 due to small numerical errors that can accumulate because of the finite precision
# MAGIC with which computers represent numbers.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC To improve readability, we can also turn off the scientific notation when printing tensor
# MAGIC values by setting sci_mode to False:
# MAGIC </div>

# COMMAND ----------

torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's now encapsulate this process in a PyTorch module that we can use in the GPT
# MAGIC model later:
# MAGIC </div>

# COMMAND ----------

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC This specific implementation of layer Normalization operates on the last dimension of the
# MAGIC input tensor x, which represents the embedding dimension (emb_dim). 
# MAGIC
# MAGIC The variable eps is a
# MAGIC small constant (epsilon) added to the variance to prevent division by zero during
# MAGIC normalization. 
# MAGIC
# MAGIC The scale and shift are two trainable parameters (of the same dimension
# MAGIC as the input) that the LLM automatically adjusts during training if it is determined that
# MAGIC doing so would improve the model's performance on its training task. 
# MAGIC
# MAGIC This allows the model
# MAGIC to learn appropriate scaling and shifting that best suit the data it is processing.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC _A small note on biased variance_

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC In our variance calculation method, we have opted for an implementation detail by
# MAGIC setting unbiased=False. 
# MAGIC
# MAGIC For those curious about what this means, in the variance
# MAGIC calculation, we divide by the number of inputs n in the variance formula. 
# MAGIC
# MAGIC This approach does not apply Bessel's correction, which typically uses n-1 instead of n in
# MAGIC the denominator to adjust for bias in sample variance estimation. 
# MAGIC
# MAGIC This decision results in a so-called biased estimate of the variance. 
# MAGIC
# MAGIC For large-scale language
# MAGIC models (LLMs), where the embedding dimension n is significantly large, the
# MAGIC difference between using n and n-1 is practically negligible. 
# MAGIC
# MAGIC We chose this approach to ensure compatibility with the GPT-2 model's normalization layers and because it
# MAGIC reflects TensorFlow's default behavior, which was used to implement the original GPT2 model.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's now try the LayerNorm module in practice and apply it to the batch input:
# MAGIC </div>

# COMMAND ----------

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see based on the results, the layer normalization code works as expected and
# MAGIC normalizes the values of each of the two inputs such that they have a mean of 0 and a
# MAGIC variance of 1:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT ARCHITECTURE PART 3: FEEDFORWARD NEURAL NETWORK WITH GELU ACTIVATION

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's implement the GELU activation function approximation used by GPT-2:
# MAGIC </div>

# COMMAND ----------

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC To get an idea of what this GELU function looks like and how it compares to the ReLU
# MAGIC function, let's plot these functions side by side:
# MAGIC </div>

# COMMAND ----------

import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# Some sample data
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see in the resulting plot, ReLU is a piecewise linear function that
# MAGIC outputs the input directly if it is positive; otherwise, it outputs zero. 
# MAGIC
# MAGIC GELU is a smooth, nonlinear function that approximates ReLU but with a non-zero gradient for negative values.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The smoothness of GELU, as shown in the above figure, can lead to better optimization properties
# MAGIC during training, as it allows for more nuanced adjustments to the model's parameters. 
# MAGIC
# MAGIC In contrast, ReLU has a sharp corner at zero, which can sometimes make optimization harder,
# MAGIC especially in networks that are very deep or have complex architectures. 
# MAGIC
# MAGIC Moreover, unlike RELU, which outputs zero for any negative input, GELU allows for a small, non-zero output
# MAGIC for negative values. 
# MAGIC
# MAGIC This characteristic means that during the training process, neurons that
# MAGIC receive negative input can still contribute to the learning process, albeit to a lesser extent
# MAGIC than positive inputs.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Next, let's use the GELU function to implement the small neural network module,
# MAGIC FeedForward, that we will be using in the LLM's transformer block later:
# MAGIC </div>

# COMMAND ----------

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# COMMAND ----------

print(GPT_CONFIG_124M["emb_dim"])


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see in the preceding code, the FeedForward module is a small neural network
# MAGIC consisting of two Linear layers and a GELU activation function. 
# MAGIC
# MAGIC In the 124 million parameter GPT model, it receives the input batches with tokens that have an embedding
# MAGIC size of 768 each via the GPT_CONFIG_124M dictionary where GPT_CONFIG_124M["emb_dim"]
# MAGIC = 768.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's use the GELU function to implement the small neural network module,
# MAGIC FeedForward, that we will be using in the LLM's transformer block later:
# MAGIC </div>

# COMMAND ----------

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768) #A
out = ffn(x)
print(out.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The FeedForward module we implemented in this section plays a crucial role in enhancing
# MAGIC the model's ability to learn from and generalize the data. 
# MAGIC
# MAGIC
# MAGIC Although the input and output dimensions of this module are the same, it internally expands the embedding dimension
# MAGIC into a higher-dimensional space through the first linear layer.
# MAGIC
# MAGIC This expansion is followed by a non-linear GELU activation, and then a contraction back to
# MAGIC the original dimension with the second linear transformation. 
# MAGIC
# MAGIC Such a design allows for the
# MAGIC exploration of a richer representation space.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Moreover, the uniformity in input and output dimensions simplifies the architecture by
# MAGIC enabling the stacking of multiple layers, as we will do later, without the need to adjust
# MAGIC dimensions between them, thus making the model more scalable.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT ARCHITECTURE PART 4: SHORTCUT CONNECTIONS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let us see how we can add shortcut connections to the forward method:
# MAGIC </div>

# COMMAND ----------

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC The code implements a deep neural network with 5 layers, each consisting of a Linear
# MAGIC layer and a GELU activation function. 
# MAGIC
# MAGIC In the forward pass, we iteratively pass the input
# MAGIC through the layers and optionally add the shortcut connections  if
# MAGIC the self.use_shortcut attribute is set to True.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's use this code to first initialize a neural network without shortcut connections. Here,
# MAGIC each layer will be initialized such that it accepts an example with 3 input values and returns
# MAGIC 3 output values. The last layer returns a single output value:
# MAGIC </div>

# COMMAND ----------

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123) # specify random seed for the initial weights for reproducibility
model_without_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Next, we implement a function that computes the gradients in the the model's backward
# MAGIC pass:
# MAGIC </div>

# COMMAND ----------

def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In the preceding code, we specify a loss function that computes how close the model output
# MAGIC and a user-specified target (here, for simplicity, the value 0) are. 
# MAGIC
# MAGIC Then, when calling loss.backward(), PyTorch computes the loss gradient for each layer in the model. 
# MAGIC
# MAGIC We can iterate through the weight parameters via model.named_parameters(). 
# MAGIC
# MAGIC Suppose we have a 3×3 weight parameter matrix for a given layer. 
# MAGIC
# MAGIC In that case, this layer will have 3×3 gradient values, and we print the mean absolute gradient of these 3×3 gradient values to
# MAGIC obtain a single gradient value per layer to compare the gradients between layers more
# MAGIC easily.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In short, the .backward() method is a convenient method in PyTorch that computes loss
# MAGIC gradients, which are required during model training, without implementing the math for the
# MAGIC gradient calculation ourselves, thereby making working with deep neural networks much
# MAGIC more accessible. 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC
# MAGIC Let's now use the print_gradients function and apply it to the model without skip
# MAGIC connections:
# MAGIC </div>

# COMMAND ----------

print_gradients(model_without_shortcut, sample_input)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC As we can see based on the output of the print_gradients function, the gradients become
# MAGIC smaller as we progress from the last layer (layers.4) to the first layer (layers.0), which
# MAGIC is a phenomenon called the vanishing gradient problem.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Let's now instantiate a model with skip connections and see how it compares:
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC As we can see, based on the output, the last layer (layers.4) still has a larger gradient
# MAGIC than the other layers. 
# MAGIC
# MAGIC However, the gradient value stabilizes as we progress towards the
# MAGIC first layer (layers.0) and doesn't shrink to a vanishingly small value.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC In conclusion, shortcut connections are important for overcoming the limitations posed
# MAGIC by the vanishing gradient problem in deep neural networks. 
# MAGIC
# MAGIC Shortcut connections are a core building block of very large models such as LLMs, and they will help facilitate more effective
# MAGIC training by ensuring consistent gradient flow across layers when we train the GPT model 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT ARCHITECTURE PART 5: CODING ATTENTION AND LINEAR LAYERS IN A TRANSFORMER BLOCK

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Let us code a transformer block as follows:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Step 1: Shortcut connection for attention block
# MAGIC
# MAGIC Step 2:  Shortcut connection for feed forward block
# MAGIC
# MAGIC Step 3: Add the original input back
# MAGIC </div>

# COMMAND ----------

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The given code defines a TransformerBlock class in PyTorch that includes a multi-head
# MAGIC attention mechanism (MultiHeadAttention) and a feed forward network (FeedForward),
# MAGIC both configured based on a provided configuration dictionary (cfg), such as
# MAGIC GPT_CONFIG_124M
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC Layer normalization (LayerNorm) is applied before each of these two components, and
# MAGIC dropout is applied after them to regularize the model and prevent overfitting. 
# MAGIC
# MAGIC This is also known as Pre-LayerNorm. 
# MAGIC
# MAGIC Older architectures, such as the original transformer model,
# MAGIC applied layer normalization after the self-attention and feed-forward networks instead,
# MAGIC known as Post-LayerNorm, which often leads to worse training dynamics.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC The class also implements the forward pass, where each component is followed by a
# MAGIC shortcut connection that adds the input of the block to its output. This critical feature helps
# MAGIC gradients flow through the network during training and improves the learning of deep
# MAGIC models 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Using the GPT_CONFIG_124M dictionary we defined earlier, let's instantiate a transformer
# MAGIC block and feed it some sample data
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Create sample input of shape [batch_size, num_tokens, emb_dim]
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(123)
x = torch.rand(2, 4, 768) #A
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see from the code output, the transformer block maintains the input dimensions
# MAGIC in its output, indicating that the transformer architecture processes sequences of data
# MAGIC without altering their shape throughout the network.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC The preservation of shape throughout the transformer block architecture is not incidental
# MAGIC but a crucial aspect of its design. 
# MAGIC
# MAGIC This design enables its effective application across a wide
# MAGIC range of sequence-to-sequence tasks, where each output vector directly corresponds to an
# MAGIC input vector, maintaining a one-to-one relationship. 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC
# MAGIC However, the output is a context vector
# MAGIC that encapsulates information from the entire input sequence.
# MAGIC
# MAGIC This means that while the physical dimensions of the sequence (length and feature size)
# MAGIC remain unchanged as it passes through the transformer block, the content of each output
# MAGIC vector is re-encoded to integrate contextual information from across the entire input
# MAGIC sequence.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT ARCHITECTURE PART 6: ENTIRE GPT MODEL ARCHITECTURE IMPLEMENTATION

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC The device setting will allow us to train the model on a CPU or GPU, depending on which device the input
# MAGIC data sits
# MAGIC     
# MAGIC </div>

# COMMAND ----------

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC The __init__ constructor of this GPTModel class initializes the token and positional
# MAGIC embedding layers using the configurations passed in via a Python dictionary, cfg. 
# MAGIC
# MAGIC These
# MAGIC embedding layers are responsible for converting input token indices into dense vectors and
# MAGIC adding positional information.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC Next, the __init__ method creates a sequential stack of TransformerBlock modules
# MAGIC equal to the number of layers specified in cfg. 
# MAGIC
# MAGIC Following the transformer blocks, a
# MAGIC LayerNorm layer is applied, standardizing the outputs from the transformer blocks to
# MAGIC stabilize the learning process. 
# MAGIC
# MAGIC Finally, a linear output head without bias is defined, which
# MAGIC projects the transformer's output into the vocabulary space of the tokenizer to generate
# MAGIC logits for each token in the vocabulary.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC The forward method takes a batch of input token indices, computes their embeddings,
# MAGIC applies the positional embeddings, passes the sequence through the transformer blocks,
# MAGIC normalizes the final output, and then computes the logits, representing the next token's
# MAGIC unnormalized probabilities. We will convert these logits into tokens and text outputs in the
# MAGIC next section.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Let's now initialize the 124 million parameter GPT model using the GPT_CONFIG_124M
# MAGIC dictionary we pass into the cfg parameter and feed it with the batch text input we created
# MAGIC at the beginning of this chapter:
# MAGIC </div>

# COMMAND ----------

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see, the output tensor has the shape [2, 4, 50257], since we passed in 2 input
# MAGIC texts with 4 tokens each. The last dimension, 50,257, corresponds to the vocabulary size of
# MAGIC the tokenizer. In the next section, we will see how to convert each of these 50,257-
# MAGIC dimensional output vectors back into tokens.
# MAGIC     
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Using the numel() method, short for "number of elements," we can collect the total
# MAGIC number of parameters in the model's parameter tensors:
# MAGIC </div>

# COMMAND ----------

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-danger">
# MAGIC Earlier, we spoke of initializing a 124
# MAGIC million parameter GPT model, so why is the actual number of parameters 163 million, as
# MAGIC shown in the preceding code output?
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC The reason is a concept called weight tying that is used in the original GPT-2
# MAGIC architecture, which means that the original GPT-2 architecture is reusing the weights from
# MAGIC the token embedding layer in its output layer. 
# MAGIC
# MAGIC To understand what this means, let's take a
# MAGIC look at the shapes of the token embedding layer and linear output layer that we initialized
# MAGIC on the model via the GPTModel earlier:
# MAGIC
# MAGIC </div>

# COMMAND ----------

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see based on the print outputs, the weight tensors for both these layers have the
# MAGIC same shape:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC The token embedding and output layers are very large due to the number of rows for the
# MAGIC 50,257 in the tokenizer's vocabulary. Let's remove the output layer parameter count from
# MAGIC the total GPT-2 model count according to the weight tying:
# MAGIC </div>

# COMMAND ----------

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see, the model is now only 124 million parameters large, matching the original
# MAGIC size of the GPT-2 model.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC Weight tying reduces the overall memory footprint and computational complexity of the
# MAGIC model. However, in my experience, using separate token embedding and output layers
# MAGIC results in better training and model performance; hence, we are using separate layers in
# MAGIC our GPTModel implementation. The same is true for modern LLMs.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Lastly, let us compute the memory requirements of the 163 million parameters in our
# MAGIC GPTModel object:
# MAGIC </div>

# COMMAND ----------

total_size_bytes = total_params * 4 #A
total_size_mb = total_size_bytes / (1024 * 1024) #B
print(f"Total size of the model: {total_size_mb:.2f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC In conclusion, by calculating the memory requirements for the 163 million parameters in
# MAGIC our GPTModel object and assuming each parameter is a 32-bit float taking up 4 bytes, we
# MAGIC find that the total size of the model amounts to 621.83 MB, illustrating the relatively large
# MAGIC storage capacity required to accommodate even relatively small LLMs.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC In this section, we implemented the GPTModel architecture and saw that it outputs
# MAGIC numeric tensors of shape [batch_size, num_tokens, vocab_size]. In the next section,
# MAGIC we will write the code to convert these output tensors into text.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT ARCHITECTURE PART 7: GENERATING TEXT FROM OUTPUT TOKENS

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Let us implement the token-generation process as follows:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC Step 1: idx is a (batch, n_tokens) array of indices in the current context
# MAGIC
# MAGIC Step 2: Crop current context if it exceeds the supported context size E.g., if LLM supports only 5 tokens, and the
# MAGIC context size is 10 then only the last 5 tokens are used as context
# MAGIC
# MAGIC Step 3: Focus only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
# MAGIC
# MAGIC Step 4: probas has shape (batch, vocab_size)
# MAGIC
# MAGIC Step 5: idx_next has shape (batch, 1)
# MAGIC
# MAGIC Step 6: Append sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
# MAGIC
# MAGIC </div>

# COMMAND ----------

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC In the preceeding code, the generate_text_simple function, we use a softmax function to
# MAGIC convert the logits into a probability distribution from which we identify the position with the
# MAGIC highest value via torch.argmax. 
# MAGIC
# MAGIC The softmax function is monotonic, meaning it preserves
# MAGIC the order of its inputs when transformed into outputs. 
# MAGIC
# MAGIC So, in practice, the softmax step is
# MAGIC redundant since the position with the highest score in the softmax output tensor is the
# MAGIC same position in the logit tensor. 
# MAGIC
# MAGIC In other words, we could apply the torch.argmax function
# MAGIC to the logits tensor directly and get identical results. 
# MAGIC
# MAGIC However, we coded the conversion to
# MAGIC illustrate the full process of transforming logits to probabilities, which can add additional
# MAGIC intuition, such as that the model generates the most likely next token, which is known as
# MAGIC greedy decoding.
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC     
# MAGIC In the next chapter, when we will implement the GPT training code, we will also
# MAGIC introduce additional sampling techniques where we modify the softmax outputs such that
# MAGIC the model doesn't always select the most likely token, which introduces variability and
# MAGIC creativity in the generated text.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Let's now try out the generate_text_simple function with the "Hello, I am" context
# MAGIC as model input
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC First, we encode the input context into token IDs:
# MAGIC </div>

# COMMAND ----------

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
print("encoded_tensor.shape:", encoded_tensor.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Next, we put the model into .eval() mode, which disables random components like
# MAGIC dropout, which are only used during training, and use the generate_text_simple function
# MAGIC on the encoded input tensor:
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC We disable dropout since we are not training the model
# MAGIC
# MAGIC </div>

# COMMAND ----------

model.eval() #A
out = generate_text_simple(
model=model,
idx=encoded_tensor,
max_new_tokens=6,
context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC Using the .decode method of the tokenizer, we can convert the IDs back into text:
# MAGIC </div>

# COMMAND ----------

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-info">
# MAGIC
# MAGIC As we can see, based on the preceding output, the model generated gibberish, which is not
# MAGIC at all coherent text. 
# MAGIC
# MAGIC What happened? 
# MAGIC
# MAGIC The reason why the model is unable to produce coherent text is that we haven't trained it yet. 
# MAGIC
# MAGIC So far, we just
# MAGIC implemented the GPT architecture and initialized a GPT model instance with initial random
# MAGIC weights.
# MAGIC
# MAGIC </div>

# COMMAND ----------


