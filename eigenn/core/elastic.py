"""
A reimplement of pymatgen elasticity, but use PyTorch tensors instead of numpy.

This only implements a minimal set of functionality.
"""
import itertools
import warnings

import torch
from torch import Tensor

voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
reverse_voigt_map = torch.tensor([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


class GeometricTensor:
    """
    A general class for geometric tensor.

    This tries to reproduce the functionality of pymatgen.core.tensors.Tensor
    """

    def __init__(self, t: Tensor, vscale: Tensor = None, check_rank=None):
        obj = t
        obj.rank = len(obj.shape)

        if check_rank and check_rank != obj.rank:
            raise ValueError(
                "{} input must be rank {}".format(obj.__class__.__name__, check_rank)
            )

        vshape = torch.Size([3] * (obj.rank % 2) + [6] * (obj.rank // 2))
        obj._vscale = torch.ones(vshape, dtype=obj.dtype, device=obj.device)
        if vscale is not None:
            obj._vscale = vscale
        if obj._vscale.shape != vshape:
            raise ValueError(
                "Voigt scaling matrix must be the shape of the "
                "voigt notation matrix or vector."
            )
        if not all(i == 3 for i in obj.shape):
            raise ValueError(
                "Pymatgen only supports 3-dimensional tensors, "
                "and default tensor constructor uses standard "
                "notation.  To construct from voigt notation, use"
                " {}.from_voigt".format(obj.__class__.__name__)
            )

        self._t = obj

    @classmethod
    def from_voigt(cls, voigt_input: Tensor):
        """
        Constructor based on the voigt notation vector or matrix.

        Args:
            voigt_input: voigt input for a given tensor
        """
        rank = sum(voigt_input.shape) // 3
        t = cls(
            torch.zeros([3] * rank, dtype=voigt_input.dtype, device=voigt_input.device)
        )
        if voigt_input.shape != t._vscale.shape:
            raise ValueError("Invalid shape for voigt matrix")
        voigt_input = voigt_input / t._vscale
        this_voigt_map = t.get_voigt_dict(rank)
        for ind, v in this_voigt_map.items():
            t._t[ind] = voigt_input[v]

        return t

    @property
    def voigt(self):
        """
        Returns the tensor in Voigt notation
        """

        v_matrix = torch.zeros(self._vscale.shape, dtype=self.dtype, device=self.device)
        this_voigt_map = self.get_voigt_dict(self.rank)
        for ind, v in this_voigt_map.items():
            v_matrix[v] = self._t[ind]
        if not self.is_voigt_symmetric():
            warnings.warn(
                "Tensor is not symmetric, information may "
                "be lost in voigt conversion."
            )

        return v_matrix * self._vscale

    def is_voigt_symmetric(self, tol=1e-6):
        """
        Tests symmetry of tensor to that necessary for voigt-conversion
        by grouping indices into pairs and constructing a sequence of
        possible permutations to be used in a tensor transpose
        """
        transpose_pieces = [[[0 for i in range(self.rank % 2)]]]
        transpose_pieces += [
            [range(j, j + 2)] for j in range(self.rank % 2, self.rank, 2)
        ]
        for n in range(self.rank % 2, len(transpose_pieces)):
            if len(transpose_pieces[n][0]) == 2:
                transpose_pieces[n] += [transpose_pieces[n][0][::-1]]
        for trans_seq in itertools.product(*transpose_pieces):
            trans_seq = list(itertools.chain(*trans_seq))
            if (self._t - torch.permute(self._t, trans_seq) > tol).any():
                return False
        return True

    @staticmethod
    def get_voigt_dict(rank):
        """
        Returns a dictionary that maps indices in the tensor to those
        in a voigt representation based on input rank

        Args:
            rank (int): Tensor rank to generate the voigt map
        """
        vdict = {}
        for ind in itertools.product(*[range(3)] * rank):
            v_ind = ind[: rank % 2]
            for j in range(rank // 2):
                pos = rank % 2 + 2 * j
                v_ind += (reverse_voigt_map[ind[pos : pos + 2]],)
            vdict[ind] = v_ind
        return vdict

    @property
    def tensor(self):
        return self._t

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def _vscale(self):
        return self._t._vscale

    @property
    def rank(self):
        return self._t.rank

    @property
    def device(self):
        return self._t.device


class ElasticTensor(GeometricTensor):
    """
    The 3x3x3x3 second-order elastic tensor, C_{ijkl}, with various methods for
    estimating other properties derived from the second order elastic tensor.


    Unlike pymatgen Tensor that subclass numpy array, this is a wrapper over pytorch
    Tensor to avoiding the bizarre stuff when subclassing pytorch tensors.
    See https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/5
    """

    def __init__(self, t: Tensor):
        """
        Args:
            t: 3x3x3x3 tensors
        """
        assert t.shape == torch.Size(
            [3, 3, 3, 3]
        ), f"Expect shape (3,3,3,3), but get {t.shape}"
        super().__init__(t)

    @property
    def compliance_tensor(self):
        """
        returns the Voigt-notation compliance tensor,
        which is the matrix inverse of the
        Voigt-notation elastic tensor
        """
        s_voigt = torch.linalg.inv(self.voigt)
        return ComplianceTensor.from_voigt(s_voigt)

    @property
    def k_voigt(self):
        """
        returns the K_v bulk modulus
        """
        return self.voigt[:3, :3].mean()

    @property
    def g_voigt(self):
        """
        returns the G_v shear modulus
        """
        return (
            2.0 * self.voigt[:3, :3].trace()
            - torch.triu(self.voigt[:3, :3]).sum()
            + 3 * self.voigt[3:, 3:].trace()
        ) / 15.0

    @property
    def k_reuss(self):
        """
        returns the K_r bulk modulus
        """
        return 1.0 / self.compliance_tensor.voigt[:3, :3].sum()

    @property
    def g_reuss(self):
        """
        returns the G_r shear modulus
        """
        return 15.0 / (
            8.0 * self.compliance_tensor.voigt[:3, :3].trace()
            - 4.0 * torch.triu(self.compliance_tensor.voigt[:3, :3]).sum()
            + 3.0 * self.compliance_tensor.voigt[3:, 3:].trace()
        )

    @property
    def k_vrh(self):
        """
        returns the K_vrh (Voigt-Reuss-Hill) average bulk modulus
        """
        return 0.5 * (self.k_voigt + self.k_reuss)

    @property
    def g_vrh(self):
        """
        returns the G_vrh (Voigt-Reuss-Hill) average shear modulus
        """
        return 0.5 * (self.g_voigt + self.g_reuss)

    @property
    def y_mod(self):
        """
        Calculates Young's modulus (in SI units) using the
        Voigt-Reuss-Hill averages of bulk and shear moduli
        """
        return 9.0e9 * self.k_vrh * self.g_vrh / (3.0 * self.k_vrh + self.g_vrh)

    @property
    def property_dict(self):
        """
        returns a dictionary of properties derived from the elastic tensor
        """
        props = [
            "k_voigt",
            "k_reuss",
            "k_vrh",
            "g_voigt",
            "g_reuss",
            "g_vrh",
            "y_mod",
        ]
        return {prop: getattr(self, prop) for prop in props}


class ComplianceTensor(GeometricTensor):
    """
    This class represents the compliance tensor, and exists
    primarily to keep the voigt-conversion scheme consistent
    since the compliance tensor has a unique vscale
    """

    def __init__(self, t: Tensor):
        """
        Args:
            t: input tensor
        """
        vscale = torch.ones((6, 6), dtype=t.dtype, device=t.device)
        vscale[3:] *= 2
        vscale[:, 3:] *= 2
        super().__init__(t, vscale=vscale)
