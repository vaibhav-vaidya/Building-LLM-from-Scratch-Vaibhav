# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC | Author  | VAIBHAV SHASHIKANT VAIDYA |
# MAGIC |---------|---------------------------|
# MAGIC | Date    | 02-01-2026                |
# MAGIC | Version | V1.0                      |
# MAGIC | Topic   | Selft Attention mechanism |

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
