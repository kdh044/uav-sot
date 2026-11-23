"""CPU checks for actual gradients, invalid observations and sampling geometry."""
import json
import torch
from .query_decoder import IdentityObservationDecoder


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = IdentityObservationDecoder(dim=32, queries=4, levels=3, points=4, layers=2)
    features = [torch.randn(2, 32, h, h, requires_grad=True) for h in (32, 16, 8)]
    refs = torch.full((2, 4, 2), .5)
    sizes = torch.full((2, 4, 2), .08)
    sigma = torch.full((2, 4, 2), .1, requires_grad=True)
    memory = torch.randn(2, 3, 32, requires_grad=True)
    before = memory.detach().clone()
    valid_memory = torch.tensor([[True, False, True], [True, True, False]])
    out = model(features, refs, sizes, sigma, memory, valid_memory)
    final = out['final']
    loss = (torch.nn.functional.cross_entropy(final['association_logits'], torch.tensor([0, 4]))
            + final['boxes_cxcywh'].square().mean()
            + torch.nn.functional.binary_cross_entropy_with_logits(final['write_logits'], torch.zeros(2, 4)))
    loss.backward()
    for x in features + [sigma, memory]:
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.equal(before, memory.detach()), 'Forward must not rewrite memory'
    assert final['association_logits'].shape == (2, 5)
    assert torch.allclose(final['sample_weights'].sum((2, 3)), torch.ones(2, 4), atol=1e-6)
    with torch.no_grad():
        zero_masks = [torch.zeros((2, f.shape[2], f.shape[3]), dtype=torch.bool) for f in features]
        empty = model(features, refs, sizes, sigma, memory, valid_memory, zero_masks)['final']
        assert not empty['sample_valid'].any()
        assert empty['sample_weights'].eq(0).all()
        assert empty['association_logits'].argmax(-1).eq(4).all()
        # First-layer detail coordinates do not widen with camera uncertainty.
        low = model(features, refs, sizes, sigma * 0, memory, valid_memory)['auxiliary'][0]
        high = out['auxiliary'][0]
        low_detail = (low['sample_locations'][:, :, :2] - refs[:, :, None, None]).abs()
        high_detail = (high['sample_locations'][:, :, :2] - refs[:, :, None, None]).abs()
        assert high_detail.max() <= sizes.max() * .5 + 1e-6
        low_context = (low['sample_locations'][:, :, -1] - refs[:, :, None]).square().sum(-1).mean()
        high_context = (high['sample_locations'][:, :, -1] - refs[:, :, None]).square().sum(-1).mean()
        assert high_context > low_context
        # Masked recent memory cannot influence output, even if its values change.
        alternate = memory.detach().clone()
        alternate[~valid_memory] = 10000
        replay = model(features, refs, sizes, sigma, alternate, valid_memory)['final']
        assert torch.allclose(replay['association_logits'], final['association_logits'], atol=1e-6)
    result = dict(status='passed', checks=['backprop_to_all_feature_levels_motion_and_memory',
                  'immutable_memory_input', 'Q_plus_null_output', 'attention_normalization',
                  'all_invalid_forces_no_match', 'detail_radius_bound',
                  'context_sampling_widens_with_motion_sigma', 'masked_memory_invariance'],
                  parameters=sum(p.numel() for p in model.parameters()),
                  scope='Untrained feature-level prototype only, not a tracking accuracy test')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
