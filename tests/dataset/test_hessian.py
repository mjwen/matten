import torch

from eigenn.dataset.hessian import separate_diagonal_blocks, symmetrize_hessian


def test_symmetrize_hessian():
    n1 = 2
    H1 = torch.arange(n1 * 3 * n1 * 3).reshape(n1 * 3, n1 * 3).to(torch.float)
    ref_sym_H1 = (H1 + H1.T) / 2

    n2 = 4
    H2 = torch.arange(n2 * 3 * n2 * 3).reshape(n2 * 3, n2 * 3).to(torch.float)
    ref_sym_H2 = (H2 + H2.T) / 2

    tmp_H1 = H1.reshape(n1, 3, n1, 3)
    tmp_H2 = H2.reshape(n2, 3, n2, 3)

    batched_H = torch.cat(
        [
            torch.swapaxes(tmp_H1, 1, 2).reshape(-1, 3, 3),
            torch.swapaxes(tmp_H2, 1, 2).reshape(-1, 3, 3),
        ]
    )

    sym_batch_H = symmetrize_hessian(batched_H, natoms=[n1, n2])

    sym_H = torch.split(sym_batch_H, [n1**2, n2**2])

    sym_H1 = sym_H[0].reshape(n1, n1, 3, 3)
    sym_H1 = torch.swapaxes(sym_H1, 1, 2).reshape(n1 * 3, n1 * 3)
    assert torch.allclose(sym_H1, ref_sym_H1)

    sym_H2 = sym_H[1].reshape(n2, n2, 3, 3)
    sym_H2 = torch.swapaxes(sym_H2, 1, 2).reshape(n2 * 3, n2 * 3)
    assert torch.allclose(sym_H2, ref_sym_H2)


def test_separate_diagonal_blocks():
    N = 4
    H = torch.arange(N * 3 * N * 3).reshape(N * 3, N * 3).to(torch.float)

    diag_blocks, off_blocks, off_diag_layout = separate_diagonal_blocks(H)

    assert diag_blocks.shape == (N, 3, 3)
    assert off_blocks.shape == (N * N - N, 3, 3)

    # diag part
    for i in range(N):
        assert torch.allclose(diag_blocks[i], H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3])

    # off diag part
    k = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                torch.allclose(diag_blocks[i], H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3])
                torch.allclose(off_diag_layout[k], torch.tensor((i, j)))
                k += 1
