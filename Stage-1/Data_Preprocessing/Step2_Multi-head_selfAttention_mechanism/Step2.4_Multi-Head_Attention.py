# Databricks notebook source
# MAGIC %pip install torch

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
