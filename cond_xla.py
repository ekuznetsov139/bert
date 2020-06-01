import tensorflow as tf

use_xla = 0

def conditional_xla():
  def decorator(func):
    if use_xla==1:
      return tf.function(experimental_compile=True,experimental_relax_shapes=True)(func)
    else:
      return func
  return decorator

