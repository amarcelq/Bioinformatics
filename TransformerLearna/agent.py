from tensorforce.agents import PPOAgent
from dataclasses import dataclass
import tensorflow as tf
from functools import partial
import numpy as np

@dataclass
class NetworkConfig:
    """
    Dataclass providing the network configuration.
    """
    vocab_size = 7  # {., (, ), A, U, G, C}
    embedding_dim = 128
    num_heads = 8
    dim_feed_forward = embedding_dim * 4
    num_encoder_blocks = 8
    dropout_rate = 0.1 
    output_units = 57

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=embedding_dim)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, embedding_dim, dim_feed_forward, dropout_rate):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dim_feed_forward, activation='relu'),
      tf.keras.layers.Dense(embedding_dim),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
  
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class EncoderBlock(tf.keras.layers.Layer):
  def __init__(self,*, embedding_dim, num_heads, dim_feed_forward, dropout_rate):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
        dropout=dropout_rate)

    self.ffn = FeedForward(embedding_dim, dim_feed_forward, dropout_rate)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

def get_network(network_config: NetworkConfig):
    input_layer = tf.keras.layers.Input(shape=(59,), dtype=tf.int32, name="input")
    x = PositionalEmbedding(network_config.vocab_size, network_config.embedding_dim)(input_layer)
    for _ in range(network_config.num_encoder_blocks):
        x = EncoderBlock(embedding_dim=network_config.embedding_dim,
                         num_heads=network_config.num_heads,
                         dim_feed_forward=network_config.dim_feed_forward,
                         dropout_rate=network_config.dropout_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last", keepdims=False)(x)

    output_layer = tf.keras.layers.Dense(
        units=network_config.output_units,
        activation="relu",
        name="output"
    )(x)

    transformer = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return transformer

@dataclass
class AgentConfig:
    """
    Dataclass providing the agent configuration.
    """
    learning_rate: float = 6.442010833400271e-05
    batch_size: int = 123
    likelihood_ratio_clipping: float = 0.3
    entropy_regularization: float = 0.00015087352506343337
    max_episode_timesteps = 600
    optimizer=dict(
        optimizer='adam', learning_rate=learning_rate, clipping_threshold=1e-2,
        multi_step=10, subsampling_fraction=64, linesearch_iterations=5,
        doublecheck_update=True
    ),


