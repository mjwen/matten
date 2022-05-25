import pytest
import torch
from pymatgen.analysis.elasticity.elastic import ComplianceTensor as PMGComplianceTensor
from pymatgen.analysis.elasticity.elastic import ElasticTensor as PMGElasticTensor
from pymatgen.core.tensors import Tensor

from eigenn.core.elastic import ComplianceTensor, ElasticTensor, GeometricTensor


@pytest.fixture(scope="session")
def tensors():
    # d = torch.arange(81).to(torch.float).reshape(3, 3, 3, 3) + 0.1
    torch.random.manual_seed(35)
    d = torch.rand(3, 3, 3, 3)
    g = GeometricTensor(d)
    t = Tensor(d.numpy())

    return g, t


def test_geometric_tensor(tensors):
    g, t = tensors
    g_v = g.voigt
    t_v = t.voigt

    # voigt
    assert torch.allclose(g_v, torch.as_tensor(t_v, dtype=g.dtype))

    # from voigt
    g2 = GeometricTensor.from_voigt(g_v)
    t2 = Tensor.from_voigt(t_v)

    assert torch.allclose(g2.tensor, torch.as_tensor(t2, dtype=g.dtype))


def test_compliance_tensor(tensors):
    g, t = tensors
    g_c = ComplianceTensor(g.tensor)
    t_c = PMGComplianceTensor(t)

    assert torch.allclose(g_c.tensor, torch.as_tensor(t_c, dtype=g.dtype))


def test_elastic_tensor(tensors):
    g, t = tensors
    g_e = ElasticTensor(g.tensor)
    t_e = PMGElasticTensor(t)

    g_prop = g_e.property_dict
    t_prop = t_e.property_dict

    for k in g_prop:
        g_v = g_prop[k]
        t_v = t_prop[k]
        assert torch.allclose(g_v, torch.as_tensor(t_v, dtype=g.dtype))
