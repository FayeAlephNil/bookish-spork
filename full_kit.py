from collections import defaultdict
from collections import Counter
import string

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional, Tuple, Callable, Dict, Any, Sequence
import heapq

import matplotlib.pyplot as plt
import matplotlib.colors

import votekit.ballot_generator.std_generator.spacial as spacial
import votekit.ballot_generator as bg
from votekit.metrics import euclidean_dist
from votekit.pref_profile import RankProfile
from votekit.elections import STV,Borda,Plurality

from display import gen_voter_display_type, prepare_candidates_for_display, display_by_type
from distortion import distortion

def alph_seq(limit):
    """
    Returns a list of strings starting from A, B, C... 
    to AA, AB, AC... up to the specified limit.
    """
    sequence = []
    letters = string.ascii_uppercase

    for i in range(limit):
        current_label = ""
        n = i
        
        # Build the label from right to left
        while n >= 0:
            current_label = letters[n % 26] + current_label
            n = (n // 26) - 1
            
        sequence.append(current_label)
    
    return sequence


def spatial_profile_from_types_profile_marked_data(
    number_of_ballots: int,
    candidates: list[str],
    voter_dist: Callable[..., np.ndarray] = np.random.uniform,
    voter_dist_kwargs: Optional[Dict[str, Any]] = None,
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

    try:
        voter_dist(**voter_dist_kwargs)
    except TypeError:
        raise TypeError("Invalid kwargs for the voter distribution.")

    if candidate_dist_kwargs is None:
        if candidate_dist is np.random.uniform:
            candidate_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
        else:
            candidate_dist_kwargs = {}

    try:
        candidate_dist(**candidate_dist_kwargs)
    except TypeError:
        raise TypeError("Invalid kwargs for the candidate distribution.")

    try:
        v = voter_dist(**voter_dist_kwargs)
        c = candidate_dist(**candidate_dist_kwargs)
        distance(v.pos, c)
    except TypeError:
        raise TypeError(
            "Distance function is invalid or incompatible "
            "with voter/candidate distributions."
        )

    candidate_position_dict = {
        c: candidate_dist(**candidate_dist_kwargs) for c in candidates
    }

    voters = np.array(
        [voter_dist(**voter_dist_kwargs) for _ in range(number_of_ballots)]
    )
    voter_positions = np.array([v.pos for v in voters])

    ballot_pool = np.full((number_of_ballots, len(candidates)), frozenset("~"))

    for i in range(number_of_ballots):
        distance_tuples = [
            (c, distance(voter_positions[i], c_position))
            for c, c_position, in candidate_position_dict.items()
        ]
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
    return (
        RankProfile(
            candidates=candidates,
            df=df,
            max_ranking_length=n_candidates,
        ),
        candidate_position_dict,
        voters
    )


def region_generator(region,num_ballots,candidate_list,candidate_dist,cand_kwargs):
    return spatial_profile_from_types_profile_marked_data(
        number_of_ballots= num_ballots,
        candidates= candidate_list,
        voter_dist=region.gen_one_random,
        candidate_dist = candidate_dist,
        candidate_dist_kwargs=cand_kwargs)

def clean_winners(wins):
    winners = []
    for cset in wins:
        for c in cset:
            winners.append(c)
    return winners

def from_region_to_display(region,election_type=STV,num_cands=20, num_winners=3,display_cands=True,show_distortion=True):
    cand_dist = np.random.uniform
    cand_kwargs = {'low': (-1,-1), 'high': (1,1), 'size': 2}
    prof, cands, data = region_generator(region,200, alph_seq(num_cands), cand_dist,cand_kwargs)

    voter_names = []
    region_names = []
    for voter in region.voters:
        voter_names.append(voter.name)
        region_names.append(voter.region)
    
    voter_display_info = gen_voter_display_type(voter_names,region_names)
    fig,ax=plt.subplots()
    
    if display_cands:
        winners = None
        if election_type != None:
            winners = clean_winners(election_type(prof,m=num_winners).get_elected())
        cands_data = prepare_candidates_for_display(cands, defaultdict(lambda: 'g'), winners=winners)
        if show_distortion:
            dist = distortion(data, cands, winners)
            display_by_type(data,cands_data,*voter_display_info,ax=ax)
            ax.text(0.95, 0.95, f'Distortion: {round(dist[0],3)}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes, # Use axes coordinates (0 to 1)
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5) # Optional: add a text box
               )
        return plt.show()
    else:
        display_by_type(data,[],*voter_display_info,ax=ax)
        return plt.show()
        
