import numpy as np
import tensorflow as tf

def xavier_weight_init():
  """
  Returns function that creates random tensor. 

  The specified function will take in a shape (tuple or 1-d array) and must
  return a random tensor of the specified shape and must be drawn from the
  Xavier initialization distribution.

  Hint: You might find tf.random_uniform useful.
  """
  def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

    Args:
      shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
      out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    ### YOUR CODE HERE
    import uuid
    name = kwargs.get("name", uuid.uuid4().hex.upper()[:6])
    dim_sum = sum(shape)
    epsilon = np.sqrt(6) / np.sqrt(dim_sum)
    out = tf.get_variable(
      name=name,
      shape=shape,
      initializer=tf.random_uniform_initializer(
        minval=-epsilon,
        maxval=epsilon
      )
    )
    ### END YOUR CODE
    return out
  # Returns defined initializer function.
  return _xavier_initializer

def test_initialization_basic():
  """
  Some simple tests for the initialization.
  """
  print("Running basic tests...")
  xavier_initializer = xavier_weight_init()
  shape = (1,)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape

  shape = (1, 2, 3)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape
  print("Basic (non-exhaustive) Xavier initialization tests pass\n")

def test_initialization():
  """ 
  Use this space to test your Xavier initialization code by running:
      python q1_initialization.py 
  This function will not be called by the autograder, nor will
  your tests be graded.
  """
  print("Running your tests...")
  ### YOUR CODE HERE
  sess = tf.Session()
  xavier_initializer = xavier_weight_init()
  shape = (4, 4)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape
  max_tensor = tf.reduce_max(tf.abs(xavier_mat))
  assert max_tensor.get_shape().ndims == 0
  sess.run(tf.initialize_all_variables())
  max_value = sess.run(max_tensor)
  xavier_max = (np.sqrt(6) / np.sqrt(8))
  print("Checking tensor: {2} -> Got a max of {0}, expected max no more than {1}".format(max_value, xavier_max, xavier_mat.name))
  assert max_value < xavier_max
  ### END YOUR CODE

if __name__ == "__main__":
    test_initialization_basic()
    test_initialization()
