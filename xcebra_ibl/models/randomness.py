"""Explicitly seed CEBRA's private sampler generators (CEBRA 0.6)."""
import torch
from cebra.distributions.base import HasGenerator


def seed_loader_generators(loader, seed):
    """Seed nested distributions, including their priors, in stable order.

    CEBRA 0.6 HasGenerator.__init__ calls generator.seed(), disregarding the
    supplied seed. Global torch.manual_seed therefore does not suffice.
    Store _seed too: CEBRA uses it when transferring generator devices.
    """
    seen, count = set(), 0

    def visit(value):
        nonlocal count
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, HasGenerator):
            value._seed = (int(seed)+count) % (2**63-1)
            value.generator.manual_seed(value._seed)
            count += 1
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])
        elif value is loader or type(value).__module__.startswith('cebra.distributions'):
            for key, item in sorted(vars(value).items()):
                if not isinstance(item, (torch.Tensor, torch.Generator)):
                    visit(item)

    visit(loader)
    return count
