import unittest
import numpy as np
import ormachine

class simple_test(unittest.TestCase):
    def test(self):
        np.random.seed(1)

        X = 2*np.array([[0,0,1,1,0,0],[1,1,0,0,0,0],[0,0,1,1,1,1]])-1
        
        X = np.concatenate(100*[X])

        orm = ormachine.machine()

        data = orm.add_matrix(val=X, sampling_indicator=False)

        layer1 = orm.add_layer(size=3, child=data, lbda_init=2, z_init=0)
                               # u_density_conditions=[1,0,0,0])

        orm.infer(convergence_window=50, no_samples=200,
                  convergence_eps=1e-3, burn_in_min=2000,
                  burn_in_max=3000)

        self.assertEqual(1/(1+np.exp(-orm.layers[0].lbda())),1)


if __name__ == '__main__':
    unittest.main()
