import torch
from e3nn.o3 import Irreps

from eigenn.data.transform import MeanNormNormalize


def test_mean_norm_normalize():
    irreps = Irreps("0e+2x1e+2e")

    N = 3
    dim = 1 + 2 * 3 + 5  # 12
    data = torch.arange(N * dim).reshape(N, dim).to(torch.float)

    mnn = MeanNormNormalize(irreps, eps=0.0)
    mean1, norm1 = mnn.compute_statistics(data)

    state_dict = mnn.state_dict()
    mean = state_dict["mean"]
    norm = state_dict["norm"]

    assert torch.allclose(mean1, mean)
    assert torch.allclose(norm1, norm)
    assert mean.shape == torch.Size((dim,))
    assert norm.shape == torch.Size((dim,))

    # check scalars
    scalars = data[:, 0]
    assert mean[0] == scalars.mean()

    # unbiased = False to not use N-1 when computing std
    assert norm[0] == torch.std(scalars, unbiased=False)

    # make sure the second 1e is correct
    # which is in columns 4~6 of data
    data2 = data[:, 4:7]
    n = data2.square().mean(dim=1).mean(dim=0).sqrt()
    assert torch.allclose(norm[4], n)
    assert torch.allclose(norm[5], n)
    assert torch.allclose(norm[6], n)

    # check forward and backward can get the same
    transformed = mnn(data)
    inversed = mnn.inverse(transformed)
    assert torch.allclose(data, inversed)
