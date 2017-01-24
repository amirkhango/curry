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
