# The OrMachine

To install and test the packge run the following commands:
```
>>> git clone https://github.com/TammoR/ormachine
>>> python3 setup.py install
>>> python3 tests/test_ormachine.py
```

## Basic usage example for a single layer Boolean Matrix Factorisation
```
import ormachine
import numpy as np

X = 2*np.array([[0,0,1,1,0,0],[1,1,0,0,0,0],[0,0,1,1,1,1]])-1
X = np.concatenate(200*[X])

orm = ormachine.machine()

data = orm.add_matrix(val=X, sampling_indicator=False)
# if sampling indicator is True, it's looking for a corresp. lbda. there is none

layer1 = orm.add_layer(size=3, child=data, lbda_init=4)
layer1.u.set_prior(.05)
# layer1.z.set_sampling_indicator(np.ones(layer1.z().shape))

# layer2 = orm.add_layer(size=2, child=data)

# layer2 = orm.add_layer(size=2, child=layer1.z, lbda_init=2)

layer3 = orm.add_layer(size=2, child=layer1.z, lbda_init=2)

orm.infer(convergence_window=50, no_samples=200, convergence_eps=1e-3, burn_in_min=500, burn_in_max=10000)
```
