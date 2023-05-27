from typing import List, Tuple

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

    def __init__(self, function_name: str, parameters: List[float]) -> None:
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

    def integral_vec(self, points: List[float]) -> List[float]:
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

    def __init__(self, primitive_type: str, tag: str, coordinates: List[float]) -> None:
        """
        The constructor of the ``PrimitiveAtom`` class.

        :param primitive_type:
        :param tag:
        :param coordinates:
        """

class LoCoHD:
    """

    :param categories: The primitive types to be used.
    :param w_func: The ``WeightFunction`` to be used.
    """

    categories: List[str]

    def __init__(self, categories: List[str], w_func: WeightFunction) -> None:
        """
        The constructor of the ``LoCoHD`` class.

        :param categories:
        :param w_func:
        """

    def from_anchors(self, seq_a: List[str], seq_b: List[str], dists_a: List[float], dists_b: List[float]) -> float:
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
                  dmx_a: List[List[float]],
                  dmx_b: List[List[float]]) -> List[float]:
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
                    coords_a: List[List[float]],
                    coords_b: List[List[float]]) -> List[float]:
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
                        only_hetero_contacts: bool,
                        threshold_distance: float) -> List[float]:
        """
        Compares two structures with a given primitive atom sequence pair. This function can be used the most
        conveniently most of the time.

        :param prim_a: The primitive atom sequence of the first structure.
        :param prim_b: The primitive atom sequence of the second structure.
        :param anchor_pairs: The index-pairs of the anchor atoms, pointing to the corresponding primitive atom pairs
            given in ``prim_a`` and ``prim_b``.
        :param only_hetero_contacts: A flag specifying whether the ``tag`` field of the ``PrimitiveAtom`` class
            should be used to ban homo-residue contacts.
        :param threshold_distance: A distance above primitive atoms are not considered inside the environment
            of an anchor atom.
        :return: The list of LoCoHD scores.
        """