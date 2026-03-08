from region import Region, make_regions, tri_party,two_bloc_weighted
from person import Person, PersonType,circle_gauss_system
from full_kit import region_generator, alph_seq, from_region_to_display, Simulation
import person

import full_kit
import numpy as np

from votekit.elections import STV,Borda,Plurality,BlocPlurality

def anomaly_add_one_person():
    seed = 857473969
    num_winners = 4
    num_cands = 20
    num_ballots = 1000
    R_sep=0.9
    people = circle_gauss_system(2,offset=np.pi/2,sigma=0.1,size=R_sep/2)
    region = Region({
        people[0]: 500,
        people[1]: 500
    })
    nation = Region.combine([region])
    
    cand_dist = lambda: nation.gen_one_random().pos
    cand_kwargs = {}
    #cand_dist=np.random.uniform
    #cand_kwargs = {'low': (-0.6,-1), 'high': (0.6,1), 'size': 2}
    sim = Simulation(nation,cand_dist=cand_dist,cand_kwargs=cand_kwargs,seed=seed)
    
    national_vote,_ = sim.run_local_votes(num_cands,num_ballots)
    national_result = sim.run_national_election(national_vote,num_winners=num_winners,election_type=Borda,interpolate=False) 
    
    sim.display(national_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1),save_at='images/add_one_person_before.png')
    worst_result = sim.worst_heur(national_result,interpolate=False)
    sim.display(worst_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1),save_at='images/add_one_person_after.png')

def anomaly_add_one_person_interp():
    seed = 857473969
    num_winners = 4
    num_cands = 20
    num_ballots = 1000
    R_sep=0.9
    people = circle_gauss_system(2,offset=np.pi/2,sigma=0.1,size=R_sep/2)
    region = Region({
        people[0]: 500,
        people[1]: 500
    })
    nation = Region.combine([region])
    
    cand_dist = lambda: nation.gen_one_random().pos
    cand_kwargs = {}
    #cand_dist=np.random.uniform
    #cand_kwargs = {'low': (-0.6,-1), 'high': (0.6,1), 'size': 2}
    sim = Simulation(nation,cand_dist=cand_dist,cand_kwargs=cand_kwargs,seed=seed)
    
    national_vote,_ = sim.run_local_votes(num_cands,num_ballots)
    national_result = sim.run_national_election(national_vote,num_winners=num_winners,election_type=Borda, interpolate=True) 
    
    sim.display(national_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1),save_at='images/add_one_person_interp_before.png')
    worst_result = sim.worst_heur(national_result,interpolate=True)
    sim.display(worst_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1), save_at='images/add_one_person_interp_after.png')
