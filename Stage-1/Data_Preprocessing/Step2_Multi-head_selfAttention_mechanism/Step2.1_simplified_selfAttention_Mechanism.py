# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC | Author  | VAIBHAV SHASHIKANT VAIDYA |
# MAGIC |---------|---------------------------|
# MAGIC | Date    | 02-01-2026                |
# MAGIC | Version | V1.0                      |
# MAGIC | Topic   | Selft Attention mechanism |

# COMMAND ----------

# MAGIC %pip install torch

# COMMAND ----------

import matplotlib.pyplot as plt

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

x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C

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
# MAGIC We now compute the context vector as a weighted sum over the value vectors.
# MAGIC
# MAGIC Here, the attention weights serve as a weighting factor that weighs the respective importance of each value vector.
# MAGIC
# MAGIC We can use matrix multiplication to obtain the output in one step:

# COMMAND ----------

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# COMMAND ----------

# MAGIC %md
# MAGIC So far, we only computed a single context vector, z(2).
# MAGIC
# MAGIC In the next section, we will generalize the code to compute all context vectors in the input sequence, z(1)to z (T)
