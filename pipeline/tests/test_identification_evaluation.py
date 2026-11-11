"""Regression checks for evaluation integrity, not model accuracy claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluate_identification import paired_iou, prediction_input, export
import torch
from region_filters import reject_vertical_option_a_corner, rank_candidates


class IdentificationEvaluationTests(unittest.TestCase):
    def test_corner_filter_is_specific_to_identity_and_location(self):
        reject=reject_vertical_option_a_corner
        self.assertTrue(reject('03a_vertical_option_a',[542,469,97,10],(640,480)))
        self.assertTrue(reject('03a_vertical_option_a',[1084,938,194,20],(1280,960)))
        self.assertFalse(reject('03a_vertical_option_a',[200,200,100,100],(640,480)))
        self.assertFalse(reject('03a_vertical_option_a',[542,100,97,10],(640,480)))
        self.assertFalse(reject('03b_vertical_option_b',[542,469,97,10],(640,480)))
        self.assertFalse(reject('03a_vertical_option_a',None,(640,480)))
        self.assertFalse(reject('03a_vertical_option_a',[.70,.70,.04,.10],(1,1)))

    def test_filtered_candidate_keeps_measured_score_but_cannot_win(self):
        def rows():
            return [{'cad_key':'03a_vertical_option_a','category_id':5,'score':.9,'semantic':.95},
                    {'cad_key':'03b_vertical_option_b','category_id':6,'score':.7,'semantic':.8}]
        filtered=rank_candidates(rows(),[542,469,97,10],(640,480))
        self.assertEqual(filtered[0]['cad_key'],'03b_vertical_option_b')
        self.assertEqual(filtered[1]['score'],.9)
        self.assertEqual(filtered[1]['raw_rank'],1)
        self.assertFalse(filtered[1]['eligible'])
        self.assertIsNone(filtered[1]['rank'])
        self.assertTrue(filtered[1]['filter_reason'])
        unfiltered=rank_candidates(rows(),[542,469,97,10],(640,480),enabled=False)
        self.assertEqual(unfiltered[0]['cad_key'],'03a_vertical_option_a')
        outside=rank_candidates(rows(),[200,200,100,100],(640,480))
        self.assertEqual(outside[0]['cad_key'],'03a_vertical_option_a')

    def test_ui_and_suite_share_the_same_scoped_filter(self):
        from sam6d_ui.engine import reject_repeated_region as ui_reject
        from run_suite_pddl import reject_repeated_region as suite_reject
        for key,bbox in [('03a_vertical_option_a',[542,469,97,10]),
                         ('03a_vertical_option_a',[200,200,100,100]),
                         ('03b_vertical_option_b',[542,469,97,10])]:
            expected=reject_vertical_option_a_corner(key,bbox,(640,480))
            self.assertEqual(ui_reject({'part_key':key,'bbox':bbox},(640,480)),expected)
            self.assertEqual(suite_reject(key,bbox,(640,480)),expected)

    def test_top_k_does_not_count_an_excluded_CAD(self):
        rows=rank_candidates([{'cad_key':'03a_vertical_option_a','category_id':5,'score':.9,'semantic':.95},
                              {'cad_key':'03b_vertical_option_b','category_id':6,'score':.7,'semantic':.8}],
                             [542,469,97,10],(640,480))
        frame={'frame_id':1,'name':'one','rgb_path':'rgb.png','expected_cad':'03a_vertical_option_a',
               'predicted_cad':'03b_vertical_option_b','correct':False,'status':'predicted','error':None,
               'scores':rows,'timing':{},'margin':None,'semantic_winner':'03b_vertical_option_b',
               'original_assignment_winner':'03b_vertical_option_b'}
        run={'catalog':[{'key':r['cad_key']} for r in rows], 'validation':{'total':1,'valid':1},
             'frames':[frame],'command':'test'}
        with tempfile.TemporaryDirectory() as t:
            summary=export(run,Path(t))
            self.assertEqual(summary['topk']['2']['matches'],0)
            saved=json.loads((Path(t)/'scores_long.json').read_text())
            excluded=next(r for r in saved if not r['eligible'])
            self.assertEqual(excluded['score'],.9)
            self.assertIsNone(excluded['rank'])

    def test_disjoint_box_does_not_zero_other_pairs(self):
        a=torch.tensor([[0.,0.,10.,10.],[20.,20.,25.,25.],[0.,0.,0.,0.]])
        b=torch.tensor([[0.,0.,10.,10.],[0.,0.,5.,5.],[0.,0.,0.,0.]])
        torch.testing.assert_close(paired_iou(a,b),torch.tensor([1.,0.,0.]))

    def test_ground_truth_and_label_fields_are_not_prediction_inputs(self):
        f={'frame_id':7,'rgb':'a.png','depth':'d.png','camera':'k.json',
           'name':'cad_a','expected_cad_key':'cad_a','filename_hint_only':'cad_a','future_ground_truth':'cad_a'}
        before=prediction_input(f)
        for k in ('name','expected_cad_key','filename_hint_only','future_ground_truth'):f[k]='cad_b'
        self.assertEqual(before,prediction_input(f))
        self.assertEqual(set(before),{'frame_id','rgb','depth','camera'})

    def test_missing_candidate_scores_are_null_and_denominators_are_selected(self):
        run={'catalog':[{'key':'a'},{'key':'b'}], 'validation':{'total':5,'valid':4},'command':'test',
             'frames':[{'frame_id':1,'name':'one','rgb_path':'a.png','expected_cad':'a','predicted_cad':None,
                        'correct':None,'status':'failed','error':'no masks','scores':[],'timing':{}}]}
        with tempfile.TemporaryDirectory() as t:
            summary=export(run,Path(t))
            self.assertEqual(summary['omitted'],4)
            self.assertIsNone(summary['top1_on_predictions'])
            self.assertEqual(summary['top1_on_selected'],0)
            rows=json.loads((Path(t)/'scores_wide.json').read_text())
            self.assertIsNone(rows[0]['a__score'])
            self.assertEqual(len(json.loads((Path(t)/'scores_long.json').read_text())),2)


if __name__=='__main__':unittest.main()
