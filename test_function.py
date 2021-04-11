# -*- coding: utf-8 -*-
"""


@author: Lucas
"""

import sys
import pandas as pd
import unittest
import numpy as np
from module_function import Embarkedint, sexint

#def fun_test(para1):
#    return para1*10



class Testfunct(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(np.array([["male", "Q"], ["male", "S"], ["female", "C"], ["female", "Q"]]), columns=['Sex', 'Embarked'])

    def test_sexint(self):
        self.assertListEqual(list(sexint(self.data["Sex"])),[1,1,0,0])

    def test_embarkedint(self):
        self.assertListEqual(list(Embarkedint(self.data["Embarked"])),[0,2,1,0])

if __name__ == '__main__':
    unittest.main()