import unittest
import numpy as np
from PIL import Image
import torch
from analyze_failure_reasons import tensor_variant, source_variant, context_predictions


class FailureReasonsTests(unittest.TestCase):
    def test_blur_spreads_impulse_and_preserves_constant(self):
        x=torch.zeros(1,1,16,16)
        x[...,8,8]=1
        mild=tensor_variant(x,'blur_0.5')
        strong=tensor_variant(x,'blur_1.0')
        self.assertGreater(mild[0,0,8,8],strong[0,0,8,8])
        self.assertGreater(strong[0,0,8,7],0)
        self.assertAlmostEqual(strong.sum().item(),1,places=6)
        torch.testing.assert_close(tensor_variant(torch.ones_like(x),'blur_1.0'),torch.ones_like(x))
        self.assertEqual(x.sum().item(),1)

    def test_resize_controls(self):
        white=Image.fromarray(np.full((240,256),255,dtype=np.uint8))
        letter=source_variant(white,'aspect_preserving')
        self.assertEqual(tuple(letter.shape),(1,224,224))
        self.assertEqual(letter[:,:7].sum().item(),0)
        self.assertEqual(letter[:,-7:].sum().item(),0)
        self.assertTrue(torch.equal(letter[:,7:-7],torch.ones(1,210,224)))
        self.assertTrue(torch.equal(source_variant(white,'source_bicubic'),torch.ones(1,224,224)))
        x=torch.ones(1,1,224,224)
        torch.testing.assert_close(tensor_variant(x,'roundtrip_216'),x)

    def test_context_excludes_target_and_sorts_within_patient(self):
        rows=[{'path':str(i),'patient_id':p,'label':y,'slice_id':s}
              for i,(p,y,s) in enumerate([('A',0,3),('B',1,9),('A',0,1),('B',1,2),('A',0,2)])]
        r=context_predictions(rows,[.9,.8,.1,.6,.2])
        self.assertAlmostEqual(r['other_slices_mean'][0],.15)
        self.assertAlmostEqual(r['neighbour_mean'][0],.55)
        self.assertAlmostEqual(r['neighbour_mean'][4],.4)
        self.assertAlmostEqual(r['other_slices_mean'][1],.6)
        self.assertAlmostEqual(r['neighbour_mean'][1],.7)
        rows[2]['label']=1
        with self.assertRaises(ValueError):
            context_predictions(rows,[.9,.8,.1,.6,.2])


if __name__=='__main__':
    unittest.main()
