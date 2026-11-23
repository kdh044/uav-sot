"""CPU functional checks that do not require SAM3 or the frozen encoder checkpoint."""
import json
import torch
from .student import RecoveryStudent
from .losses import tracking_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(43)
    model=RecoveryStudent(dim=32,layers=2,points=4,load_context=False)
    args=dict(current=torch.randn(1,3,64,80),previous=torch.randn(1,3,64,80),
              template_detail=torch.randn(1,3,128,128),template_context=torch.randn(1,3,128,128),
              prior_box=torch.tensor([[.5,.5,.2,.2]]),context_features=torch.randn(1,768,16,16))
    output=model(**args)
    assert output['final']['boxes_cxcywh'].shape==(1,8,4)
    gt=torch.tensor([[.5,.5,.2,.2]])
    yy,xx=torch.meshgrid(torch.linspace(0,1,16),torch.linspace(0,1,20),indexing='ij')
    teacher_mask=torch.exp(-((xx-.5).square()+(yy-.5).square())/.02)[None,None].requires_grad_()
    # A controlled, nonuniform teacher field tests coordinate supervision separately
    # from whether random untrained proposals happen to cover the object.
    locations=torch.empty_like(output['final']['sample_locations']).uniform_(.42,.58).requires_grad_()
    output['final']['sample_locations']=locations
    loss,metrics=tracking_loss(output,gt,args['prior_box'],gt.clone(),teacher_mask,
                              kd_weight=.3,observation_kd=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert teacher_mask.grad is None,'Teacher must be detached'
    assert locations.grad is not None and locations.grad.abs().sum()>0
    for module in [model.detail,model.context_projection,model.motion,model.decoder]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    with torch.no_grad():
        no_teacher=model(**args)
        _,empty=tracking_loss(no_teacher,gt,args['prior_box'],torch.zeros_like(gt),teacher_mask,
                              kd_weight=.3,observation_kd=True)
        assert empty['teacher_accepted']==0
        assert all(v==0 for k,v in empty.items() if k.startswith('kd_'))
        _,negative=tracking_loss(no_teacher,gt,args['prior_box'],gt,teacher_mask,
                                 kd_weight=.3,target_present=torch.tensor([False]))
        assert negative['teacher_accepted']==0 and negative['box_0']==0
        assert negative['identity_0']>0 and torch.isfinite(torch.tensor(negative['total']))
    knowledge=RecoveryStudent(dim=32,layers=1,points=4,load_context=False,knowledge_distillation=True)
    output_knowledge=knowledge(**args)
    pointer=torch.randn(1,256,requires_grad=True)
    anchor_pointer=torch.randn(1,256,requires_grad=True)
    memory_descriptor=torch.randn(1,64,requires_grad=True)
    feature_loss,feature_metrics=tracking_loss(output_knowledge,gt,args['prior_box'],gt,teacher_mask,
        kd_weight=.3,teacher_pointer=pointer,teacher_anchor_pointer=anchor_pointer,
        teacher_memory_descriptor=memory_descriptor)
    feature_loss.backward()
    assert pointer.grad is None and anchor_pointer.grad is None and memory_descriptor.grad is None
    assert knowledge.identity_transfer[0].weight.grad.abs().sum()>0
    assert knowledge.memory_transfer[0].weight.grad.abs().sum()>0
    knowledge.eval()
    with torch.no_grad():
        assert 'distilled_queries' not in knowledge(**args),'Distillation heads must be inactive at inference'
    print(json.dumps(dict(status='passed',checks=['image_to_box_forward','native_detail_and_context_gradients',
          'motion_and_decoder_gradients','teacher_detached','observation_coordinate_gradients',
          'empty_teacher_skips_all_KD','constructed_negative_supervises_no_match',
          'object_pointer_and_memory_adapter_gradients','teacher_representations_detached',
          'distillation_heads_inactive_at_inference'],
          scope='Functional prototype checks, not an accuracy test',metrics=metrics,feature_metrics=feature_metrics),indent=2))


if __name__=='__main__':main()
