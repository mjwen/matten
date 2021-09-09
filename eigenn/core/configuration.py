"""
This is copied from kliff.dataset.configuration with the extxyz support removed.
We copy it here just to make this repo not dependant on kliff. Eventually, we should
merge this back to kliff.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from pymatgen.core.structure import Structure

from eigenn.typing import PBC


class Configuration:
    r"""
    Base class of an atomic configuration (for both crystals and molecules).

    This is used to store the information of an atomic configuration, e.g. super cell,
    species, coords, energy, forces...

    Args:
        species: A list of N strings giving the species of the atoms, where N is the
            number of atoms.
        coords: A Nx3 matrix of the coordinates of the atoms, where N is the number of
            atoms.
        energy: energy of the configuration.
        forces: A Nx3 matrix of the forces on atoms, where N is the number of atoms.
        stress: A list with 6 components in Voigt notation, i.e. it returns
            :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
            \sigma_{xy}]`. See: https://en.wikipedia.org/wiki/Voigt_notation
        cell: A 3x3 matrix of the lattice vectors. The first, second, and third rows are
            :math:`a_1`, :math:`a_2`, and :math:`a_3`, respectively.
        pbc: A list with 3 components indicating whether periodic boundary condition
            is used along the directions of the first, second, and third lattice vectors.
        weight: weight of the configuration in the loss function.
        identifier: a (unique) identifier of the configuration
        kwargs: extra property of a configuration (e.g. charge of each atom). The
            property can be accessed as an attribute.
    """

    def __init__(
        self,
        species: List[str],
        coords: np.ndarray,
        *,
        energy: float = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[List[float]] = None,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[PBC] = None,
        weight: float = 1.0,
        identifier: Optional[str] = None,
        **kwargs,
    ):
        self._species = species
        self._coords = coords
        self._energy = energy
        self._forces = forces
        self._stress = stress
        self._cell = cell
        self._pbc = pbc
        self._weight = weight
        self._identifier = identifier
        self._path = None

        self._kwargs = kwargs
        for key, value in kwargs.items():
            self[key] = value

    @classmethod
    def from_pymatgen_structure(cls, struct: Structure, **kwargs):
        """
        Create a configuration from pymatgen structure.

        Args:
            struct: pymatgen `Structure`
            kwargs: extra property of a configuration (e.g. charge of each atom). The
                property can be accessed as an attribute.
        """
        cell = struct.lattice
        species = struct.species
        coords = struct.cart_coords
        PBC = [True, True, True]

        return cls(species, coords, cell=cell, PBC=PBC)

    @property
    def cell(self) -> np.ndarray:
        """
        3x3 matrix of the lattice vectors of the configuration.
        """
        return self._cell

    @property
    def pbc(self) -> PBC:
        """
        A list with 3 components indicating whether periodic boundary condition
        is used along the directions of the first, second, and third lattice vectors.
        """
        return self._pbc

    @property
    def species(self) -> List[str]:
        """
        Species string of all atoms.
        """
        return self._species

    @property
    def coords(self) -> np.ndarray:
        """
        A Nx3 matrix of the Cartesian coordinates of all atoms.
        """
        return self._coords

    @property
    def energy(self) -> Union[float, None]:
        """
        Potential energy of the configuration.
        """
        if self._energy is None:
            raise ConfigurationError("Configuration does not contain energy.")
        return self._energy

    @property
    def forces(self) -> np.ndarray:
        """
        Return a `Nx3` matrix of the forces on each atoms.
        """
        if self._forces is None:
            raise ConfigurationError("Configuration does not contain forces.")
        return self._forces

    @property
    def stress(self) -> List[float]:
        r"""
        Stress of the configuration.

        The stress is given in Voigt notation i.e
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`.

        """
        if self._stress is None:
            raise ConfigurationError("Configuration does not contain stress.")
        return self._stress

    @property
    def weight(self):
        """
        Get the weight of the configuration if the loss function.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: float):
        """
        Set the weight of the configuration if the loss function.
        """
        self._weight = weight

    @property
    def identifier(self) -> str:
        """
        Return identifier of the configuration.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str):
        """
        Set the identifier of the configuration.
        """
        self._identifier = identifier

    @property
    def path(self) -> Union[Path, None]:
        """
        Return the path of the file containing the configuration. If the configuration
        is not read from a file, return None.
        """
        return self._path

    @property
    def properties(self) -> List[str]:
        """
        Property names provided as ``kwargs`` at instantiation.
        """
        return list(self._kwargs.keys())

    def get_num_atoms(self) -> int:
        """
        Return the total number of atoms in the configuration.
        """
        return len(self.species)

    def get_num_atoms_by_species(self) -> Dict[str, int]:
        """
        Return a dictionary of the number of atoms with each species.
        """
        return self.count_atoms_by_species()

    def get_volume(self) -> float:
        """
        Return volume of the configuration.
        """
        return abs(np.dot(np.cross(self.cell[0], self.cell[1]), self.cell[2]))

    def count_atoms_by_species(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Count the number of atoms by species.

        Args:
            symbols: species to count the occurrence. If `None`, all species present
                in the configuration are used.

        Returns:
            {specie, count}: with `key` the species string, and `value` the number of
                atoms with each species.
        """

        unique, counts = np.unique(self.species, return_counts=True)
        symbols = unique if symbols is None else symbols

        natoms_by_species = dict()
        for s in symbols:
            if s in unique:
                natoms_by_species[s] = counts[list(unique).index(s)]
            else:
                natoms_by_species[s] = 0

        return natoms_by_species

    def order_by_species(self):
        """
        Order the atoms according to the species such that atoms with the same species
        have contiguous indices.
        """
        if self.forces is not None:
            species, coords, forces = zip(
                *sorted(
                    zip(self.species, self.coords, self.forces),
                    key=lambda pair: pair[0],
                )
            )
            self._species = np.asarray(species).tolist()
            self._coords = np.asarray(coords)
            self._forces = np.asarray(forces)
        else:
            species, coords = zip(
                *sorted(zip(self.species, self.coords), key=lambda pair: pair[0])
            )
            self._species = np.asarray(species)
            self._coords = np.asarray(coords)

    def __getitem__(self, key):
        """
        Gets the data of the attribute :obj:`key`.
        """
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """
        Sets the attribute :obj:`key` to :obj:`value`.
        """
        setattr(self, key, value)


class ConfigurationError(Exception):
    def __init__(self, msg):
        super(ConfigurationError, self).__init__(msg)
        self.msg = msg
