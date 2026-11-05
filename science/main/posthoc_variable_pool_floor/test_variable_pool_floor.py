#!/usr/bin/env python3
import itertools
from fractions import Fraction
import unittest
from variable_pool_floor import evaluate


class MixedPoolBounds(unittest.TestCase):
    def test_unequal_pools_exhaustively(self):
        # All pairs of sizes 1..4, output counts and reference counts.
        cases=0
        for n1,n2 in itertools.product(range(1,5),repeat=2):
            for a1,b1,a2,b2 in itertools.product(range(n1+1),range(n1+1),range(n2+1),range(n2+1)):
                pools=[(n1,a1,b1),(n2,a2,b2)];result=evaluate(pools)
                feasible=[]
                for s1,s2 in itertools.product(range(n1+1),range(n2+1)):
                    ma=Fraction(abs(a1-s1),2*n1)+Fraction(abs(a2-s2),2*n2)
                    mb=Fraction(abs(b1-s1),2*n1)+Fraction(abs(b2-s2),2*n2)
                    self.assertGreaterEqual(max(ma,mb),result['floor'])
                    feasible.append(max(ma,mb));cases+=1
                self.assertGreaterEqual(result['witness_mae'],min(feasible))
                self.assertLessEqual(result['witness_mae'],result['floor']+result['guaranteed_witness_slack'])
                r1,r2=result['witness_reference_counts']
                self.assertTrue(0<=r1<=n1 and 0<=r2<=n2)
        self.assertEqual(cases,50176)

    def test_lower_bound_can_be_nontight(self):
        result=evaluate([(1,0,1)])
        self.assertEqual(result['floor'],Fraction(1,2))
        self.assertEqual(result['witness_mae'],Fraction(1,1))

    def test_invalid_inputs(self):
        for data in ([],[(0,0,0)],[(2,-1,1)],[(2,3,0)],[(2,1.0,1)],[(2,1)]):
            with self.assertRaises(ValueError):evaluate(data)


if __name__=='__main__':unittest.main()
