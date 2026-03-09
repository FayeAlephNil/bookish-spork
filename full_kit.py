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
import measurements

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

def region_generator(region,num_ballots,candidate_list,candidate_dist,cand_kwargs,dist=euclidean_dist):
    return spatial_profile_from_types_profile_marked_data(
        number_of_ballots= num_ballots,
        candidates= candidate_list,
        voter_dist=region.gen_random,
        candidate_dist = candidate_dist,
        candidate_dist_kwargs=cand_kwargs,
        distance=dist)

def clean_winners(wins):
    winners = []
    for cset in wins:
        for c in cset:
            winners.append(c)
    return winners

def from_region_to_display(region,election_type=STV,num_cands=20, num_winners=3,display_cands=True,show_distortion=True):
    cand_dist = np.random.uniform
    cand_kwargs = {'low': (-1,-1), 'high': (1,1), 'size': 2}
    prof, cands, data,cost_array = region_generator(region,200, alph_seq(num_cands), cand_dist,cand_kwargs)

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
            dist,_ = distortion(data, cands, winners)
            display_by_type(data,cands_data,*voter_display_info,ax=ax)
            ax.text(0.95, 0.95, f'Distortion: {round(dist,3)}',
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

def det_cand_dist(cands):
    it = iter(cands)
    def dist():
        return next(it)
    return dist
        
class Simulation:
    def __init__(self, region,cand_dist=np.random.uniform,cand_kwargs=None,seed=None,dist=euclidean_dist):
        if seed == None:
            self.seed = np.random.get_state()[1][0]
            np.random.seed(self.seed)
        else:
            self.seed = seed
            np.random.seed(self.seed)
        if cand_kwargs == None:
            cand_kwargs = {'low': (-1,-1), 'high': (1,1), 'size': 2}
        self.region = region
        self.voter_names = []
        self.region_names = []
        for voter in self.region.voters:
            self.voter_names.append(voter.name)
            self.region_names.append(voter.region)
        self.cand_dist = cand_dist
        self.cand_kwargs = cand_kwargs
        self.dist = dist

    def run_national_vote_named_cands(self, cand_names, num_ballots):
        prof, cands, data,cost_array = region_generator(self.region, num_ballots, cand_names, self.cand_dist,self.cand_kwargs)
        return {
            'profile': prof,
            'candidates': cands,
            'voters': data,
            'cost_array': cost_array
        }

    def run_national_vote(self, num_cands, num_ballots):
        prof, cands, data,cst_arr = region_generator(self.region, num_ballots, alph_seq(num_cands), self.cand_dist,self.cand_kwargs,dist=self.dist)
        return {
            'profile': prof,
            'candidates': cands,
            'voters': data,
            'cost_array': cst_arr
        }

    def run_national_election(self, vote, num_winners=3,election_type=STV,interpolate=False):
        prof = vote['profile']
        cands = vote['candidates']
        voters = vote['voters']
        winners = clean_winners(election_type(prof,m=num_winners,tiebreak='random').get_elected())
        elec = {}
        elec['winners'] = winners
        elec['candidates'] = cands
        elec['voters'] = voters
        elec['cost_array'] = vote['cost_array']
        global_dist = self.global_distortion(elec,interpolate=interpolate)

        group_dists = {}
        for name in self.voter_names:
            dist = self.distortion_name(elec, names=[name],interpolate=interpolate)
            group_dists[name] = dist

        return {
            'profile': prof,
            'candidates': cands,
            'voters': voters,
            'winners': winners,
            'distortion': global_dist,
            'group_dists': group_dists,
            'cost_array': vote['cost_array']
        }

    def run_local_votes(self, num_cands, num_ballots,dist=None):
        if dist is None:
            dist = self.dist
        simulations = [(r.population/self.region.population, Simulation(r,self.cand_dist,self.cand_kwargs,dist=self.dist)) for r in self.region.subregions]
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
        cost_array = None

        for _,_,vote in locals:
            cands = cands | vote['candidates']
            voters = np.concatenate((voters, vote['voters']))

        true_num_cands = len(cands.keys())
        true_num_ballots = len(voters)

        if len(self.region.subregions) == 1:
            cost_array = locals[0][2]['cost_array']
            prof = locals[0][2]['profile']
        else:
            cost_array = np.zeros((len(cands.keys()), len(voters)))
            cand_names = cands.keys()
            ballot_pool = np.full((true_num_ballots, true_num_cands), frozenset("~"))
            for j, v in enumerate(voters):
                with_names = []
                for i, c_name in enumerate(cand_names):
                    cost_array[i,j] = dist(v.pos, cands[c_name])
                    with_names.append((c_name,cost_array[i,j]))

                candidate_ranking = np.array(
                    [frozenset({t[0]}) for t in sorted(with_names, key=lambda x: x[1])]
                )
                ballot_pool[j] = candidate_ranking
            df = pd.DataFrame(ballot_pool)
            df.index.name = "Ballot Index"
            df.columns = [f"Ranking_{i + 1}" for i in range(true_num_cands)]
            df["Weight"] = 1
            df["Voter Set"] = [frozenset()] * len(df)
            prof = RankProfile(candidates=cand_names, df=df, max_ranking_length=true_num_cands)

        national = {
            'profile': prof,
            'candidates': cands,
            'voters': voters,
            'cost_array': cost_array
        }

        return national, locals
    
    def run_local_elections(self, local_votes, num_winners,election_type=STV,interpolate=False):
            local_elecs = [(sim, sim.run_national_election(vote, int(np.floor(num_winners*prop)),interpolate=interpolate, 
                election_type=election_type) )
                for prop, sim,vote in local_votes]
            
            cands = {}
            winners = []
            voters = np.array([])
            for _,x in local_elecs:

                cands = cands | x['candidates']
                winners += x['winners']
                voters = np.concatenate((voters,x['voters']))
            if len(self.region.subregions) == 1:
                cost_array = local_elecs[0][1]['cost_array']
            else:
                cost_array = np.zeros((len(cands.keys()), len(voters)))
                cand_names = cands.keys()
                for j, v in enumerate(voters):
                    for i, c_name in enumerate(cands.keys()):
                        cost_array[i,j] = self.dist(v.pos, cands[c_name])

            elec = {}
            elec['voters'] = voters
            elec['winners'] = winners
            elec['candidates'] = cands
            elec['cost_array'] = cost_array

            group_dists = {}
            for name in self.voter_names:
                dist = self.distortion_name(elec, names=[name],interpolate=interpolate)
                group_dists[name] = dist

            global_dist = self.global_distortion(elec,interpolate=interpolate)

            return {
                'locals': local_elecs,
                'group_dists': group_dists,
                'candidates': cands,
                'voters': voters,
                'winners': winners,
                'distortion': global_dist ,
                'region': self.region,
                'cost_array': cost_array
            }


    def display(self, result,display_cands=True,show_distortion=True, show_group_dist=True,opacity=0.1,xlim=None,ylim=None,save_at=None):
        winners = result['winners']
        data = result['voters']
        cands = result['candidates']
        voter_names = []
        for voter in data:
            if not(voter.name in voter_names):
                voter_names.append(voter.name)
        voter_display_info = gen_voter_display_type(voter_names,self.region_names)
        _, colors_long_names = voter_display_info
        colors = {}
        for v in voter_names:
            colors[v] = colors_long_names[self.region_names[0]+'.'+v]
        for v in data:
            if v.name == "WORST":
                colors_long_names[v.identifier] = 'k'

        fig,ax=plt.subplots()
        if xlim != None:
            plt.xlim(*xlim)
        if ylim != None:
            plt.ylim(*ylim)
    
        if display_cands:
            cands_data = prepare_candidates_for_display(cands, defaultdict(lambda: 'g'), winners=winners)
            display_by_type(data,cands_data,*voter_display_info,ax=ax,opacity=opacity)
            if show_distortion:
                dist = result['distortion']
                ax.text(0.95, 0.95, f'Distortion: {round(dist,3)}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes, # Use axes coordinates (0 to 1)
                fontsize=10,
                #bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5) # Optional: add a text box
                )
            if show_group_dist:
                i = 1
                siz = 0.05
                for name,group_dist in result['group_dists'].items():
                    ax.text(0.95, 0.95-i*siz, f'Distortion: {round(group_dist,3)}',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes, # Use axes coordinates (0 to 1)
                    fontsize=10,
                    color=colors[name]
                    #bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5) # Optional: add a text box
                    )
                    i+=1
        else:
            display_by_type(data,[],*voter_display_info,ax=ax,opacity=opacity)
        if save_at != None:
            plt.savefig(save_at, bbox_inches='tight', dpi=300)
        return plt.show()

    def distortion_name(self,elec,names,dist=None,interpolate=False):
        if dist == None:
            dist = self.dist
        subset = []
        for v in elec['voters']:
            if v.name in names:
                subset.append(v)
        return self.subset_distortion(elec,subset,dist=dist,interpolate=interpolate)

    def global_distortion(self,elec,dist=None,interpolate=False):
        if dist == None:
            dist = self.dist
        cst_arr, winner_idxs = self.prepare_for_meas(elec,dist=dist)
        voter_labels = np.zeros(len(elec['voters']))
        return measurements.group_inefficiency(cst_arr,winner_idxs,voter_labels, 0,interpolate=interpolate)

    def subset_distortion(self, elec, subset, dist=None,interpolate=False):
        if dist == None:
            dist = self.dist
        cst_arr, winner_idxs = self.prepare_for_meas(elec,dist=dist)
        voter_labels = np.zeros(len(elec['voters']))
        for i,v in enumerate(elec['voters']):
            if v in subset:
                voter_labels[i] = 1
        return measurements.group_inefficiency(cst_arr,winner_idxs,voter_labels, 1,interpolate=interpolate)

    def idx_distortion(self,elec,idxs,dist=None,interpolate=False):
        if dist == None:
            dist = self.dist
        cst_arr, winner_idxs = self.prepare_for_meas(elec,dist=dist)
        voter_labels = np.zeros(len(elec['voters']))
        voter_labels[idxs] = 1
        return measurements.group_inefficiency(cst_arr,winner_idxs,voter_labels, 1,interpolate=interpolate)


    def prepare_for_meas(self,elec,dist=euclidean_dist):
        voters = [v.copy() for v in elec['voters']]
        cands = elec['candidates']
        winners = elec['winners']

        #cost_array = np.zeros((len(cands.keys()), len(voters)))
        #cand_names = cands.keys()
        winner_idxs = [i for i, val in enumerate(cands) if val in winners]
        #for j, v in enumerate(voters):
            #for i, c_name in enumerate(cands.keys()):
                #cost_array[i,j] = dist(v.pos, cands[c_name])
        return elec['cost_array'],winner_idxs

    def worst_random(self,elec,n_samples=100,dist=None,interpolate=False):
        if dist == None:
            dist = self.dist
        voters = [v.copy() for v in elec['voters']]
        cands = elec['candidates']
        winners = elec['winners']
        cst_arr, winner_idxs = self.prepare_for_meas(elec,dist=dist)
        distortion_val, worst_bloc_idxs = measurements.worst_random_group_inefficiency(n_samples, cst_arr,winner_idxs,interpolate=interpolate)
        worst_bloc = [voters[int(i)] for i in worst_bloc_idxs]
        for v in voters:
            if v in worst_bloc:
                v.change_name("WORST")
            else:
                v.change_name("NOT_WORST")

        return {
            'voters': voters,
            'candidates': cands,
            'winners': winners,
            'distortion': distortion_val,
            'cost_array': elec['cost_array']
        }

    def worst_heur(self, elec, dist=euclidean_dist,interpolate=False):
        voters = [v.copy() for v in elec['voters']]
        cands = elec['candidates']
        winners = elec['winners']
        cost_array, winner_idxs = self.prepare_for_meas(elec,dist=dist)
        worst_bloc_idxs = measurements.heuristic_worst_bloc(cost_array,winner_idxs,interpolate=interpolate)
        worst_bloc = [voters[int(i)] for i in worst_bloc_idxs]
        distortion_val = self.idx_distortion(elec, idxs=worst_bloc_idxs,interpolate=interpolate,dist=dist)

        for v in worst_bloc:
            v.change_name("WORST")
            v.color = 'k'
        return {
            'voters': voters,
            'candidates': cands,
            'winners': winners,
            'distortion': distortion_val,
            'group_dists': {},
            'cost_array': elec['cost_array']
        }




