from numpy import ndarray
from typing import List, Tuple, Optional, Dict, Any, Union

VectorLike = Union[List[float], ndarray]
MatrixLike = Union[List[List[float]], ndarray]

class WeightFunction:
    """
    A class used to specify the weight function inside the LoCoHD algorithm.

    :param parameters: The parameters of the weight function. These have to align
        well with the type of the weight function.
    :param function_name: The type of the weight function. Currently, the hyper exponential
        PDF (``"hyper_exp"``), the Dagum distribution PDF (``"dagum"``), the uniform distribution PDF
        (``"uniform"``), and the Kumaraswamy distribution PDF (``"kumaraswamy"``) are available.
    """

    parameters: List[float]
    function_name: str

    def __init__(self, function_name: str, parameters: VectorLike) -> None:
        """
        The constructor of the ``WeightFunction`` class.

        :param function_name:
        :param parameters:
        """

    def integral_point(self, point: float) -> float:
        """
        Gets the integral of the weight function from 0 to a given point, i.e. the
        value CDF(x).

        :param point: The upper bound of the integral.
        :return: The integral.
        """

    def integral_vec(self, points: VectorLike) -> List[float]:
        """
        Gets the integral of the weight function from 0 to all the given points, i.e. the
        values CDF(x[0]), CDF(x[1])... CDF(x[N]).

        :param points: The points at which the integrals are evaluated.
        :return: The list of integrals.
        """

    def integral_range(self, point_from: float, point_to: float) -> float:
        """
        Calculates the integral within the given bounds, i.e. the value for
        CDF(x_end) - CDF(x_start).

        :param point_from: The lower bound of the integral.
        :param point_to: The upper bound of the integral.
        :return: The integral.
        """

class PrimitiveAtom:
    """
    A class used to store a primitive atom for the LoCoHD calculation.

    :param primitive_type: The primitive type of the atom.
    :param tag: A string, that is used to decide wether a primitive atom is inside the
        environment of the anchor atom or not. For example, the only_hetero_contacts switch
        for the LoCoHD function ``from_primitives``.
    :param coordinates: The coordinates of the primitive atom.
    """

    primitive_type: str
    tag: str
    coordinates: List[float]

    def __init__(self, primitive_type: str, tag: str, coordinates: VectorLike) -> None:
        """
        The constructor of the ``PrimitiveAtom`` class.

        :param primitive_type:
        :param tag:
        :param coordinates:
        """

class TagPairingRule:
    """
    A class used to define what tag-pairings are allowed during the LoCoHD calculation.
    With this it is possible to exclude primitive atoms from an environment based on the
    ``tag`` field of the anchor atom and the ``tag`` field of a test primitive atom (near
    the anchor).
    """

    def __init__(self, variant: Dict[str, Any]) -> None:
        """
        The constructor for the ``TagPairingRule`` class. The dictionary passed to the constructor
        sets the exact tag-pairing rule. Currently, the following variants are available:

        - A dictionary with one ``"accept_same"`` key having a boolean value. If true, then primitive
          atoms with the same ``tag`` field as the anchor atom will be accepted. If false, these primitive
          atoms won't be considered as the environment of the anchor atom. This can be used to exclude
          homo-residue contacts if the ``tag`` field contains a residue-specific unique string for the
          source residue of the primitive atom.
        - A dictionary with the following keys: ``"tag_pairs"``, ``"accepted_pairs"``, ``"ordered"`` and value
          types of ``Set[Tuple[str, str]]``, ``bool``, ``bool``, respectively. The key ``"tag_pairs"`` specifies
          pairs of strings. If ``"accepted_pairs"`` is true, then these are the only tag-pairs that are accepted
          during the calculations. If it's false, then every pair is accepted except these. If ``"ordered"`` is
          true, then the first element of the pairs specify the tag of the anchor atom, while the second element
          specifies the tag of the test primitive atom. If it's false, then the order of the tuples doesn't matter,
          i.e. both tuple-orders are tested.

        :param variant: A dictionary setting the tag-pair acceptance rule. For variants see the description
          above.
        """

    def pair_accepted(self, pair: Tuple[str, str]) -> bool:
        """
        Tests whether a tag-pair is accepted or not.
        
        :param pair: The tag-pair to be tested.
        """

    def get_dbg_str(self) -> str:
        """
        Prints out the string representing the rule.
        """

class LoCoHD:
    """
    The main class used to perform the LoCoHD calculations.

    :param categories: The primitive types to be used.
    :param w_func: The ``WeightFunction`` to be used.
    :param tag_pairing_rule: The ``TagPairingRule`` employed.
    """

    categories: List[str]

    def __init__(self, 
                categories: List[str], 
                w_func: Optional[WeightFunction] = None, 
                tag_pairing_rule: Optional[TagPairingRule] = None,
                n_of_threads: Optional[int] = None) -> None:
        """
        The constructor of the ``LoCoHD`` class.

        :param categories: The full set of the primitive types that can occur during calculations. When a primitive
          type that is not part of this list is encountered, the software throws an error.
        :param w_func: Either ``None`` or a  ``WeightFunction`` instance. The weight function used inside the integral. 
          It weights the cumulative contribution of the different primitive atoms within certain distances from 
          the anchor atom, i.e. the contribution of primitive atoms within the anchor atom's environment.
          When ``None``, it defaults to the ``WeightFunction("uniform", [3., 10.])`` case.
        :param tag_pairing_rule: Either ``None`` or a  ``TagPairingRule`` instance. See the description of the 
          ``TagPairingRule`` class for what it does. When ``None``, it defaults to the
          ``TagPairingRule({"accept_same": True})`` case.
        :param n_of_threads: Either ``None`` or an integer. The number of threads the instance is allowed to use.
          When ``None``, all the threads become available for the instance.
        """

    def from_anchors(self, seq_a: List[str], seq_b: List[str], dists_a: VectorLike, dists_b: VectorLike) -> float:
        """
        Performs one LoCoHD calculation step on two environments. These environments are defined by the
        primitive types they contain (in seq_a and seq_b) and by the distances measured from the central
        anchor atom (dists_a and dists_b). The distances must be in an increasing order (this is NOT checked
        by the function!) and have to start with a distance of 0 as their first value. The corresponding seq
        and dists parameters have to have the same lengths. These mean that the first primitive types in seq_a
        and seq_b must be the primitive types of the anchor atoms.

        :param seq_a: The primitive type sequence for the first environment.
        :param seq_b: The primitive type sequence for the second environment.
        :param dists_a: The distances measured from the anchor atom for the first environment.
        :param dists_b: The distances measured from the anchor atom for the second environment.
        :return: The calculated LoCoHD score.
        """

    def from_dmxs(self,
                  seq_a: List[str],
                  seq_b: List[str],
                  dmx_a: MatrixLike,
                  dmx_b: MatrixLike) -> List[float]:
        """
        Performs LoCoHD calculations on the primitive atom distance matrices provided. Every row represents an
        (unordered) environment distance vector for a primitive atom. Also, every primitive atom is used as an
        anchor atom in this function, thus the result is a list of LoCoHD scores. This list has as many elements
        as many rows the distance matrices have (the matrices have to have the same length).

        :param seq_a: The primitive type sequence for the first structure.
        :param seq_b: The primitive type sequence for the second structure.
        :param dmx_a: The distance matrix of the first structure.
        :param dmx_b: The distance matrix of the second structure.
        :return: The list of LoCoHD scores.
        """

    def from_coords(self,
                    seq_a: List[str],
                    seq_b: List[str],
                    coords_a: MatrixLike,
                    coords_b: MatrixLike) -> List[float]:
        """
        Performs LoCoHD calculations on the primitive atom coordinates provided. It calculates the distance matrices
        with the L2 (Euclidean) metric.

        :param seq_a: The primitive type sequence for the first structure.
        :param seq_b: The primitive type sequence for the second structure.
        :param coords_a: The coordinate sequence for the first structure.
        :param coords_b: The coordinate sequence for the second structure.
        :return: The list of LoCoHD scores.
        """

    def from_primitives(self,
                        prim_a: List[PrimitiveAtom],
                        prim_b: List[PrimitiveAtom],
                        anchor_pairs: List[Tuple[int, int]],
                        threshold_distance: float) -> List[float]:
        """
        Compares two structures with a given primitive atom sequence pair. This function can be used the most
        conveniently most of the time.

        :param prim_a: The primitive atom sequence of the first structure.
        :param prim_b: The primitive atom sequence of the second structure.
        :param anchor_pairs: The index-pairs of the anchor atoms, pointing to the corresponding primitive atom pairs
            given in ``prim_a`` and ``prim_b``.
        :param threshold_distance: A distance above primitive atoms are not considered inside the environment
            of an anchor atom.
        :return: The list of LoCoHD scores.
        """