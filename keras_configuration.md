In the latest version of Keras, the default *backend* is *tensorflow* and *image_dim_ordering* is *tf*. You need to change them. 

Please set configuration as follows. 

# configuration file of Keras: ~/.keras/keras.json
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}

Ref: <https://keras.io/backend/>

In our experiment, we only tested on `Keras 1.2` and `Theano 0.9`. If you use Keras 2.x or TensorFlow, please see the following discussions: 

1. [ValueError: Only layers of same output shape can be merged using sum mode](https://github.com/lucktroy/DeepST/issues/1)
2. [Only layers of same output shape can be merged using sum mode. Layer shapes: [(None, 6, 32, 2), (None, 2, 32, 2), (None, 2, 32, 2)]](https://github.com/lucktroy/DeepST/issues/6)
