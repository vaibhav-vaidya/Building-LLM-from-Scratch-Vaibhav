# Databricks notebook source
# MAGIC %pip install torch

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

# MAGIC %run ./Step2.1_simplified_selfAttention_Mechanism

# COMMAND ----------

# MAGIC %run ./Step2.2_selfAttention_Mechanism

# COMMAND ----------

import torch

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
