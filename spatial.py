from collections import defaultdict
from collections import Counter
import string

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional, Tuple, Callable, Dict, Any, Sequence
import heapq

import votekit.ballot_generator.std_generator.spacial as spacial
import votekit.ballot_generator as bg
from votekit.metrics import euclidean_dist
from votekit.pref_profile import RankProfile
from votekit.elections import STV,Borda,Plurality

def profile_from_positions(candidate_pos_dict, voter_pos_arr, distance=euclidean_dist):
    candidates = candidate_pos_dict.keys()
    num_ballots = len(voter_pos_arr)
    num_cands = len(candidates)
    ballot_pool = np.full((num_ballots, num_cands), frozenset("~"))
    cost_array = np.zeros((num_cands,num_ballots))
    for i, pos in enumerate(voter_pos_arr):
        distance_tuples = [
            (c, distance(pos, c_position))
            for c, c_position, in candidate_pos_dict.items()
        ]
        cost_array[:,i] = np.array([d for _,d in distance_tuples])
        candidate_ranking = np.array(
            [frozenset({t[0]}) for t in sorted(distance_tuples, key=lambda x: x[1])]
        )
        ballot_pool[i] = candidate_ranking
    
    n_candidates = len(candidates)
    df = pd.DataFrame(ballot_pool)
    df.index.name = "Ballot Index"
    df.columns = [f"Ranking_{i + 1}" for i in range(n_candidates)]
    df["Weight"] = 1
    df["Voter Set"] = [frozenset()] * len(df)

    return RankProfile(
            candidates=candidates,
            df=df,
            max_ranking_length=n_candidates,
        ), cost_array

def spatial_profile_from_types_profile_marked_data(
    number_of_ballots: int,
    candidates: list[str],
    voter_dist: Callable[..., np.ndarray] = np.random.uniform,
    voter_dist_kwargs: Optional[Dict[str, Any]] = {},
    candidate_dist: Callable[..., np.ndarray] = np.random.uniform,
    candidate_dist_kwargs: Optional[Dict[str, Any]] = None,
    distance: Callable[[np.ndarray, np.ndarray], float] = euclidean_dist,
) -> Tuple[RankProfile, dict[str, np.ndarray], np.ndarray]:
    """
    Samples a metric position for number_of_ballots voters from
    the voter distribution. Samples a metric position for each candidate
    from the input candidate distribution. With sampled
    positions, this method then creates a ranked RankProfile in which
    voter's preferences are consistent with their distances to the candidates
    in the metric space.

    Args:
        number_of_ballots (int): The number of ballots to generate.
        by_bloc (bool): Dummy variable from parent class.

    Returns:
        Tuple[RankProfile, dict[str, numpy.ndarray], numpy.ndarray]:
            A tuple containing the preference profile object,
            a dictionary with each candidate's position in the metric
            space, and a matrix where each row is a single voter's position
            in the metric space.
    """
    if voter_dist_kwargs is None:
        if voter_dist is np.random.uniform:
            voter_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
        else:
            voter_dist_kwargs = {}
    voter_dist_kwargs_new  = voter_dist_kwargs | {'num_samples': number_of_ballots}

#    try:
#        voter_dist(1, **voter_dist_kwargs)
#    except TypeError:
#        raise TypeError("Invalid kwargs for the voter distribution.")

    if candidate_dist_kwargs is None:
        if candidate_dist is np.random.uniform:
            candidate_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
        else:
            candidate_dist_kwargs = {}

#    try:
#        candidate_dist(**candidate_dist_kwargs)
#    except TypeError:
#        raise TypeError("Invalid kwargs for the candidate distribution.")
#
#    try:
#        v = voter_dist(1, **voter_dist_kwargs)[0]
#        c = candidate_dist(**candidate_dist_kwargs)
#        distance(v.pos, c)
#    except TypeError:
#        raise TypeError(
#            "Distance function is invalid or incompatible "
#            "with voter/candidate distributions."
#        )
#
    candidate_position_dict = {
        c: candidate_dist(**candidate_dist_kwargs) for c in candidates
    }

    voters = voter_dist(**voter_dist_kwargs_new)
    voter_positions = np.array([v.pos for v in voters])

    prof,cost_array = profile_from_positions(candidate_position_dict, voter_positions,distance=distance)
    return (
        prof,
        candidate_position_dict,
        voters,
        cost_array
    )
