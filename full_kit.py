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
from spatial import spatial_profile_from_types_profile_marked_data

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
        
class Simulation:
    def __init__(self, region):
        self.region = region
        self.voter_names = []
        self.region_names = []
        for voter in self.region.voters:
            self.voter_names.append(voter.name)
            self.region_names.append(voter.region)

    def run_national_vote_named_cands(self, cand_names, num_ballots):
        cand_dist = np.random.uniform
        cand_kwargs = {'low': (-1,-1), 'high': (1,1), 'size': 2}
        prof, cands, data = region_generator(self.region, num_ballots, cand_names, cand_dist,cand_kwargs)
        return {
            'profile': prof,
            'candidates': cands,
            'voters': data
        }

    def run_national_vote(self, num_cands, num_ballots):
        cand_dist = np.random.uniform
        cand_kwargs = {'low': (-1,-1), 'high': (1,1), 'size': 2}
        prof, cands, data = region_generator(self.region, num_ballots, alph_seq(num_cands), cand_dist,cand_kwargs)
        return {
            'profile': prof,
            'candidates': cands,
            'voters': data
        }

    def run_national_election(self, vote, num_winners=3,election_type=STV):
        prof = vote['profile']
        cands = vote['candidates']
        voters = vote['voters']
        winners = clean_winners(election_type(prof,m=num_winners).get_elected())
        dist = distortion(voters, cands, winners)

        return {
            'profile': prof,
            'candidates': cands,
            'voters': voters,
            'winners': winners,
            'distortion': dist
        }

    def run_local_votes(self, num_cands, num_ballots):
        simulations = [(r.population/self.region.population, Simulation(r)) for r in self.region.subregions]
        cand_names = alph_seq(num_cands)
        locals = []
        idx = 0

        for prop,sim in simulations:
            local_num_ballots = int(np.floor(num_ballots*prop))
            local_num_cands = int(np.floor(num_cands*prop))
            local_cand_names = cand_names[idx:idx+local_num_cands+1]
            idx += local_num_cands+1
            local_vote = sim.run_national_vote_named_cands(local_cand_names,local_num_ballots)
            locals.append((prop,sim,local_vote))

        prof = None
        cands = {}
        voters = np.array([])
        for _,_,vote in locals:
            if prof == None:
                prof = vote['profile']
            else:
                prof = prof + vote['profile']
            cands = cands | vote['candidates']
            voters = np.concatenate((voters, vote['voters']))
        
        national = {
            'profile': prof,
            'candidates': cands,
            'voters': voters
        }

        return national, locals
    
    def run_local_elections(self, local_votes, num_winners,election_type=STV):
            local_elecs = [(sim, sim.run_national_election(vote, int(np.floor(num_winners*prop)), 
                election_type=election_type) )
                for prop, sim,vote in local_votes]
            
            cands = {}
            winners = []
            data = np.array([])
            for _,x in local_elecs:

                cands = cands | x['candidates']
                winners += x['winners']
                data = np.concatenate((data,x['voters']))

            group_dists = {}
            for name in self.voter_names:
                dist = distortion(data, cands, winners, group_name=name)
                group_dists[name] = dist

            global_dist = distortion(data,cands,winners)

            return {
                'locals': local_elecs,
                'global_dist' : global_dist,
                'group_dists': group_dists,
                'candidates': cands,
                'voters': data,
                'winners': winners,
                'distortion': global_dist ,
                'region': self.region
            }

    def display(self, result,display_cands=True,show_distortion=True):
        winners = result['winners']
        data = result['voters']
        cands = result['candidates']
        voter_display_info = gen_voter_display_type(self.voter_names,self.region_names)
        _, colors_long_names = voter_display_info
        colors = {}
        for v in self.voter_names:
            colors[v] = colors_long_names[self.region_names[0]+'.'+v]

        fig,ax=plt.subplots()
    
        if display_cands:
            cands_data = prepare_candidates_for_display(cands, defaultdict(lambda: 'g'), winners=winners)
            display_by_type(data,cands_data,*voter_display_info,ax=ax)
            if show_distortion:
                dist = result['distortion']
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


