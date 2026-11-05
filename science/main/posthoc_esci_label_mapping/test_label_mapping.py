#!/usr/bin/env python3
import tempfile
from pathlib import Path
import unittest
from analyze_label_mapping import keyed, quality, run


class MappingTest(unittest.TestCase):
    def test_known_predictions(self):
        rows=[{'items':[{'grade':'E'},{'grade':'S'},{'grade':'I'}]}]
        pred=[['Y','N','Y']]
        es,_=quality(rows,pred,{'E','S'}); exact,_=quality(rows,pred,{'E'})
        self.assertEqual((es['tp'],es['fp'],es['tn'],es['fn']),(1,1,0,1))
        self.assertAlmostEqual(es['f1'],.5); self.assertAlmostEqual(es['balanced_accuracy'],.25)
        self.assertEqual(es['ratio_mae'],0.)
        self.assertEqual((exact['tp'],exact['fp'],exact['tn'],exact['fn']),(1,1,1,0))
        self.assertAlmostEqual(exact['f1'],2/3); self.assertAlmostEqual(exact['balanced_accuracy'],.75)
        self.assertAlmostEqual(exact['ratio_mae'],1/3)

    def test_keyed_format(self):
        self.assertEqual(keyed('{"1":"Y","2":"N"}',2),['Y','N'])
        for text in ('{"1":"Y","1":"N"}', '{"2":"N","1":"Y"}', '{"1":"Y"}', '{"1":"Y","2":"X"}'):
            with self.assertRaises(ValueError): keyed(text,2)

    def test_missing_pools_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'esci_selective_holdout_selection').mkdir()
            (root/'esci_selective_holdout_selection/inputs.jsonl').write_text('')
            with self.assertRaises(ValueError): run(root)


if __name__=='__main__': unittest.main()
