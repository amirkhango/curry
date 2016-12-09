*Tested on `Windows Server 2012 R2`.*

1. Install [**DeepST**](https://github.com/lucktroy/DeepST)

2. Download [**BikeNYC**](https://github.com/lucktroy/DeepST/tree/master/data/BikeNYC) data

3. Reproduce the result of ST-ResNet 

    ```
    THEANO_FLAGS="device=gpu,floatX=float32" python exptBikeNYC.py
    ```