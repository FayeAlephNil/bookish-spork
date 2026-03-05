import numpy as np
import heapq
import random

def L1_distance_to_candidate(voting_data, cand_position):
    #cand_position = cand[2]
    voter_positions = [ballot.pos for ballot in voting_data]
    voter_relative_positions = (np.stack(voter_positions) - cand_position)
    distance = np.abs(voter_relative_positions).sum()
    return distance
    

def distortion(voting_data, cands, winner_set, group_name='',voter_subset=None, interpolate=True):
    if voter_subset == None:
        voter_subset = voting_data
    if group_name != '' or len(voter_subset) != len(voting_data):
        old_voting_data_len = len(voting_data)
        voting_data = voter_subset
        if group_name != '':
            voting_data = [voter for voter in voting_data if voter.name == group_name]
        if interpolate==True:
            group_winner_set_size = np.array([int(np.floor(len(winner_set) * len(voting_data)/old_voting_data_len)),int(np.ceil(len(winner_set) * len(voting_data)/old_voting_data_len))])
            if group_winner_set_size[1] == 0:
                return 1, []
        else:
            group_winner_set_size = int(np.floor(len(winner_set) * len(voting_data)/old_voting_data_len))
            if group_winner_set_size == 0:
                return 1, []
        group_proportion = len(winner_set) * len(voting_data)/old_voting_data_len
    else:
        group_proportion = 1
        if interpolate == False:
            group_winner_set_size = len(winner_set)
        else:
            group_winner_set_size = [len(winner_set),len(winner_set)]

    
    winner_dict = {k: cands[k] for k in winner_set if k in cands}
    L1_distances = {}
    for cand,pos in cands.items():
        L1_distances[cand] = L1_distance_to_candidate(voting_data,pos)
        
    L1_distances_winners = {k:L1_distances[k] for k in L1_distances if k in winner_dict.keys()}
        
    if interpolate == False:
        optimal_cands = dict(heapq.nsmallest(group_winner_set_size, L1_distances.items(), key=lambda kv: kv[1]))
        optimal_winners = dict(heapq.nsmallest(group_winner_set_size, L1_distances_winners.items(), key=lambda kv: kv[1]))

        optimal_distance = sum(L1_distances[k] for k in list(optimal_cands.keys()))
        winner_distance = sum(L1_distances[k] for k  in list(optimal_winners.keys()))

        return distortion, optimal_cands


    else:
        optimal_cands = [dict(heapq.nsmallest(group_winner_set_size[0], L1_distances.items(), key=lambda kv: kv[1])),dict(heapq.nsmallest(group_winner_set_size[1], L1_distances.items(), key=lambda kv: kv[1]))]
        optimal_winners = [dict(heapq.nsmallest(group_winner_set_size[0], L1_distances_winners.items(), key=lambda kv: kv[1])),dict(heapq.nsmallest(group_winner_set_size[1], L1_distances_winners.items(), key=lambda kv: kv[1]))]
        optimal_distance = [sum(L1_distances[k] for k in list(optimal_cands[0].keys())),sum(L1_distances[k] for k in list(optimal_cands[1].keys()))]
        winner_distance = [sum(L1_distances[k] for k  in list(optimal_winners[0].keys())),sum(L1_distances[k] for k  in list(optimal_winners[1].keys()))]

        distortion_floor=1
        distortion_ceil=1
        if optimal_distance[0] != 0:
            distortion_floor = winner_distance[0] / optimal_distance[0]

        if optimal_distance[1] != 0:
            distortion_ceil = winner_distance[1] / optimal_distance[1]

        distortion = [distortion_floor, distortion_ceil]
        distortion_interpolated = distortion[0] + (group_proportion-np.floor(group_proportion))*(distortion[1]-distortion[0])

        return distortion_interpolated, optimal_cands


    
        


def find_worst_group_heuristic(voting_data, cands, winner_set, trials):
    voter_diameter = max([np.linalg.norm(voter.pos) for voter in voting_data])
    worst_group_sofar = [voting_data,1]

    for i in range(0,trials):
        center = random.choice(voting_data).pos
        radius = random.uniform(0,voter_diameter)
        voters_in_circle = [voter for voter in voting_data if np.linalg.norm(voter.pos-center) < radius]
        random_group_distortion = distortion(voting_data, cands, winner_set,'', voters_in_circle)[0]
        if random_group_distortion > worst_group_sofar[1]:
            worst_group_sofar = [voters_in_circle, random_group_distortion,center, radius]
    
    return worst_group_sofar

