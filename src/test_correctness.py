import unittest
import numpy as np
import torch
from models import ContextBatchNorm2d,make_model,configure
from conformal import quantile,calibrate,measure
from run import contexts

class Checks(unittest.TestCase):
    def test_bn_matches_pytorch(self):
        torch.manual_seed(1);a=ContextBatchNorm2d(3);a.eval();a.strength=1.;a.group_size=4
        x=torch.randn(12,3,7,7)
        expected=torch.cat([torch.nn.functional.batch_norm(z,None,None,a.weight,a.bias,True,0,a.eps) for z in x.split(4)])
        torch.testing.assert_close(a(x),expected,atol=1e-6,rtol=1e-5)
    def test_context_group_vectorization(self):
        torch.manual_seed(2);m=make_model('resnet_bn',1,2);x=torch.randn(5,1,28,28);pool=torch.randn(30,1,28,28);ix=np.arange(30).reshape(2,15)
        p=contexts(m,x,pool,ix,'target',query_block=5)
        q=contexts(m,x,pool,ix,'target',query_block=1)
        np.testing.assert_allclose(p,q,atol=2e-6,rtol=1e-5)
    def test_source_invariance(self):
        torch.manual_seed(3);m=make_model('cnn_bn',1,2);x=torch.randn(4,1,28,28);pool=torch.randn(30,1,28,28);ix=np.arange(30).reshape(2,15)
        p=contexts(m,x,pool,ix,'source');np.testing.assert_array_equal(p[0],p[1])
    def test_direct_companion_independence_controls(self):
        # Bypass the source-inference broadcast shortcut and execute full batches.
        for arch in ['cnn_bn','cnn_gn']:
            torch.manual_seed(12);m=make_model(arch,1,2);configure(m,'source')
            x=torch.randn(1,1,28,28);a=torch.randn(15,1,28,28);b=torch.randn(15,1,28,28)*3+2
            with torch.inference_mode():
                expected=m(x);pa=m(torch.cat([x,a]))[:1];pb=m(torch.cat([x,b]))[:1]
            torch.testing.assert_close(expected,pa,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(expected,pb,atol=2e-6,rtol=2e-5)
    def test_mixture_variance(self):
        a=ContextBatchNorm2d(1);a.eval();a.strength=.5;a.group_size=2
        a.running_mean.fill_(0);a.running_var.fill_(1)
        x=torch.tensor([2.,4.]).reshape(2,1,1,1)
        # target mean3,var1; mixed mean1.5,var3.25
        expected=(x-1.5)/np.sqrt(3.25+a.eps)
        torch.testing.assert_close(a(x),expected)
    def test_rank_edge(self):
        self.assertTrue(np.isinf(quantile(np.arange(8),.1)));self.assertEqual(quantile(np.arange(9),.1),8)
        self.assertEqual(quantile(np.arange(19),.1),17)
    def test_aps_score_order_and_exact_endpoint(self):
        from score_sensitivity import aps_scores
        p=np.array([[[.1,.7,.2],[.55,.3,.15]]],dtype=np.float32)
        s=aps_scores(p)
        np.testing.assert_allclose(s[0,0],[1.,.7,.9],atol=2e-8)
        self.assertEqual(s[0,0,0],1.)
        self.assertEqual(s[0,1,2],1.)
        randomized=aps_scores(p,np.array([.5,.5]))
        np.testing.assert_allclose(randomized[0,0],[.95,.35,.8],atol=2e-8)
    def test_identical_contexts(self):
        p=np.random.default_rng(0).dirichlet(np.ones(3),size=100);p=np.stack([p,p,p]);y=np.arange(100)%3
        t=calibrate(p,y);np.testing.assert_array_equal(t['envelope'],t['nominal'])
    def test_event_equivalence(self):
        rng=np.random.default_rng(1);p=rng.dirichlet(np.ones(3),size=(5,200));y=rng.integers(0,3,200)
        q=calibrate(p[:,:100],y[:100])['envelope'];r=measure(p[:,100:],y[100:],q)
        worst=(1-p[:,np.arange(100,200),y[100:]]).max(0)
        self.assertAlmostEqual(r['simultaneous_coverage'],float((worst<=q[0]).mean()))
    def test_corrected_maxrank_boundary(self):
        # One context: corrected max-rank uses one more order statistic.
        p=np.array([[[1-i/100,i/100] for i in range(1,20)]])
        y=np.zeros(19,dtype=int);t=calibrate(p,y,.1)
        self.assertAlmostEqual(t['nominal'][0],.18)
        self.assertAlmostEqual(t['corrected_maxrank'][0],.19)
    def test_paired_dedup(self):
        from run import retained_indices
        z={f'{a}_images':np.array(b,dtype=np.uint8).reshape(-1,1,1) for a,b in [('train',[1,2,2]),('val',[2,3]),('test',[3,4,4])]}
        keep=retained_indices(z)
        np.testing.assert_array_equal(keep['train'],[0,1])
        np.testing.assert_array_equal(keep['val'],[1])
        np.testing.assert_array_equal(keep['test'],[1])
    def test_rank_coverage_exhaustive(self):
        # For any n+1 distinct exchangeable envelope scores, leave each out in turn.
        for n in range(9,80):
            z=np.random.default_rng(n).random(n+1); covered=0
            for i in range(n+1):covered+=int(z[i]<=quantile(np.delete(z,i),.1))
            self.assertGreaterEqual(covered/(n+1),.9-1e-12)
    def test_simultaneous_corrections_leave_one_out(self):
        # Enumerate every possible held-out index of an exchangeable finite pool.
        # Repeated/tied scores deliberately exercise rank-to-score boundaries.
        for n in [19,39,79]:
            for m in [1,3,12]:
                rng=np.random.default_rng(n+m)
                scores=np.round(rng.random((m,n+1)),2)
                p=np.stack([1-scores,scores],axis=-1);y=np.zeros(n+1,dtype=int)
                covered={k:0 for k in ['bonferroni','corrected_maxrank','envelope']}
                for i in range(n+1):
                    cal=np.delete(np.arange(n+1),i);t=calibrate(p[:,cal],y[cal])
                    for k in covered:covered[k]+=int(np.all(scores[:,i]<=t[k]))
                for k in covered:self.assertGreaterEqual(covered[k]/(n+1),.9-1e-12,msg=(n,m,k))
    def test_resnet18_source_bn_replacement(self):
        from torchvision.models import resnet18
        from run_highres import replace_bn
        torch.manual_seed(21);m=resnet18(weights=None).eval();x=torch.randn(3,3,64,64)
        with torch.inference_mode():expected=m(x)
        replace_bn(m);configure(m,'source')
        with torch.inference_mode():actual=m(x)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    def test_resnet18_target_group_equivalence(self):
        import copy
        from torchvision.models import resnet18
        from run_highres import replace_bn
        torch.manual_seed(22);stock=resnet18(weights=None).eval();target=copy.deepcopy(stock)
        for layer in stock.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.train()
        replace_bn(target);configure(target,'target',4);x=torch.randn(8,3,64,64)
        with torch.inference_mode():
            expected=torch.cat([stock(v) for v in x.split(4)]);actual=target(x)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)

class FusedTests(unittest.TestCase):
    def test_fused_target_equivalence(self):
        torch.manual_seed(23)
        layer=ContextBatchNorm2d(5).eval();layer.strength=1.;layer.group_size=4
        x=torch.randn(12,5,9,9)
        with torch.inference_mode():
            expected=layer(x);layer.fused_target=True;actual=layer(x)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()
