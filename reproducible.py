from region import Region, make_regions, tri_party,two_bloc_weighted
from person import Person, PersonType,circle_gauss_system
from full_kit import region_generator, alph_seq, from_region_to_display, Simulation
import person

import full_kit
import numpy as np
import display
import matplotlib.pyplot as plt

from votekit.elections import STV,Borda,Plurality,BlocPlurality

def anomaly_add_one_person(show_it,seeded):
    seed = None
    if seeded:
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
    blue_bloc_size = len([v for v in national_vote['voters'] if v.pos[1] < 0])
    print(f"Number in Blue block add one: {blue_bloc_size}")

    display.display(national_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1),save_at='images/add_one_person_before.png',show_it=show_it)
    worst_result = sim.worst_heur(national_result,interpolate=False)
    display.display(worst_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1),save_at='images/add_one_person_after.png',show_it=show_it)

def anomaly_add_one_person_interp(show_it,seeded):
    seed = None
    if seeded:
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
    
    display.display(national_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1),save_at='images/add_one_person_interp_before.png',show_it=show_it)
    worst_result = sim.worst_heur(national_result,interpolate=True)
    display.display(worst_result,opacity=0.1,xlim=(-1,1),ylim=(-1,1), save_at='images/add_one_person_interp_after.png',show_it=show_it)

def centrist_swing(show_it,seeded):
    to_run = []
    if seeded:
        to_run = [
            (4004134264, Borda,True, 'interp--center_swing_borda',False),
           (4004134264, STV,True, 'interp--center_swing_stv',False),
           (4004134264, Borda,False, 'nointerp--center_swing_borda',False),
           (4004134264, STV,False, 'nointerp--center_swing_stv',False),
           (3597092709, Borda, True, 'interp--center_split_borda',False),
           (3597092709, Borda, False, 'nointerp--center_split_borda',False),
           (3597092709, STV, False, 'nointerp--center_split_stv',False),
           (3597092709, STV, True, 'interp--center_split_stv',False)
        ]
    else:
        to_run = [
           (None, Borda,True, 'interp--center_swing_borda',False),
           (None, STV,True, 'interp--center_swing_stv',False),
           (None, Borda,False, 'nointerp--center_swing_borda',False),
           (None, STV,False, 'nointerp--center_swing_stv',False),
        ]
    percent_center = 0.10
    total = 1000
    num_cands = 20
    num_winners = num_cands/5
    elec_type = Borda
    interp = True
    
    parties = person.centrists(sigma=0.1,size=0.6)
    cands_lsts = []
    region = Region({
        parties['Up']: int((1-percent_center)/2 * total),
        parties['Center']: int(percent_center*total),
        parties['Down']: int((1-percent_center)/2*total)
    },deterministic=True)

    region_average = Region({
        parties['Up']: 365,
        parties['Center']: 20,
        parties['Down']: 365
    },deterministic=True)
    region_third = Region({
        parties['Up']: 85,
        parties['Center']: 80,
        parties['Down']: 85
    },deterministic=True)
    nation = Region.combine([region],deterministic=True)
    nation_reg = Region.combine([region_average,region_third],deterministic=True)
    
    for seed,elec_type,interp,file_name,unif in to_run:
        cand_dist = None
        cand_kwargs = None
        if unif:
            cand_dist = np.random.uniform
            cand_kwargs = {'low': (-0.4,-1), 'high': (0.4,1), 'size': 2}
        else:
            cand_dist = lambda: nation.gen_one_random().pos
            cand_kwargs = {}
        sim = Simulation(nation,cand_dist=cand_dist,cand_kwargs=cand_kwargs,seed=seed)

        national_vote,_ = sim.run_local_votes(num_cands,total)
        same_cands = full_kit.det_cand_dist(national_vote['candidates'].values())
        sim_loc = Simulation(nation_reg,cand_dist=same_cands,cand_kwargs={},seed=seed)

        _,local_votes = sim_loc.run_local_votes(num_cands,total)

        national_result = sim.run_national_election(national_vote,num_winners=num_winners,election_type=elec_type,interpolate=interp) 
        local_results = sim_loc.run_local_elections(local_votes,num_winners=num_winners,election_type=elec_type,interpolate=interp) 
        worst_result = sim.worst_heur(national_result,interpolate=interp)
        worst_local_result = sim.worst_heur(local_results,interpolate=interp)
        
        file = 'images/national/nat_default_' + file_name + '.png'
        file_worst = 'images/national/nat_worst_' + file_name + '.png'
        file_worst_local = 'images/local/loc_worst_' + file_name + '.png'
        file_local = 'images/local/loc_default_' + '' + file_name + '.png'
        display.display(national_result,xlim=(-1,1),ylim=(-1,1),save_at=file,show_it=show_it)
        display.display(worst_result,xlim=(-1,1),ylim=(-1,1),save_at=file_worst,show_it=show_it)
        display.display(worst_local_result,xlim=(-1,1),ylim=(-1,1),save_at=file_worst_local,show_it=show_it)
        display.display(local_results,xlim=(-1,1),ylim=(-1,1),save_at=file_local,show_it=show_it)


def fringe_parties(show_it, seeded):
    seed = None
    if seeded:
        seed = 710024847
    polarized_small = 0.05
    polarized_large = 0.35
    total = 1000
    num_cands = 50
    num_winners = 4#num_cands/5
    interp = True
    
    parties = person.fringes(sigma=0.1,size=0.6)
    region_small_polar = Region({
        parties['ExtremeUp']: int(polarized_small * total),
        parties['Up']: int(np.round((1-2*polarized_small)/2 * total)),
        parties['Down']: int(np.round((1-2*polarized_small)/2 * total)),
        parties['ExtremeDown']: int(polarized_small * total)
    },deterministic=True,name='Center')
    region_large_polar = Region({
        parties['ExtremeUp']: int(polarized_large * total),
        parties['Up']: int(np.round((1-2*polarized_large)/2 * total)),
        parties['Down']: int(np.round((1-2*polarized_large)/2 * total)),
        parties['ExtremeDown']: int(polarized_large * total)
    },deterministic=True,marker='o',name='Polar')
    nation = Region.combine([region_small_polar,region_large_polar],deterministic=True)
    region_small_polar.color_dict = {
        parties['Up']: (1,0,0),
        parties['ExtremeUp']: (1,0.7,0),
        parties['ExtremeDown']: (0,0.7,1),
        parties['Down']: (0,0,1)
    }
    
    assert region_large_polar.population == region_small_polar.population
    region_large_polar.color_dict = region_small_polar.color_dict
    
    
    
    cand_dist = lambda: nation.gen_one_random().pos
    cand_kwargs = {}
    #cand_dist=np.random.uniform
    #cand_kwargs = {'low': (-0.4,-1.5), 'high': (0.4,1.5), 'size': 2}
    sim = Simulation(nation,cand_dist=cand_dist,cand_kwargs=cand_kwargs)
    #sim = Simulation(nation,cand_dist=cand_dist,cand_kwargs=cand_kwargs,seed=710024847)
    
    for v,amt in nation.voters.items():
        print(f"{v.name} Population: {amt}")
    
    national_vote,local_votes = sim.run_local_votes(num_cands,total)
    for elec_type,name in [(Plurality,'plurality'), (STV,'stv'), (Borda,'borda')]:
        national_result = sim.run_national_election(national_vote,num_winners=num_winners,election_type=elec_type,interpolate=interp) 
        loc_result = sim.run_local_elections(local_votes,num_winners=num_winners,election_type=elec_type,interpolate=interp) 
        display.display(national_result,xlim=(-1,1),ylim=(-1.6,1.6),ax_kwargs={
            'show_cand_names': False
        }, save_at='images/fringes/fringe_national_' + name + '.png',show_it=show_it)
        display.display(loc_result,xlim=(-1,1),ylim=(-1.6,1.6),ax_kwargs={
            'show_cand_names': False
        },save_at='images/fringes/fringe_local_' + name + '.png',show_it=show_it)
        display.display(loc_result['locals'][0][1],xlim=(-1,1),ylim=(-1.6,1.6),ax_kwargs={
            'show_cand_names': False
        },save_at='images/fringes/fringe_center_region_' + name + '.png', show_it=show_it)
        display.display(loc_result['locals'][1][1],xlim=(-1,1),ylim=(-1.6,1.6),ax_kwargs={
            'show_cand_names': False
        },save_at='images/fringes/fringe_polarized_region_' + name + '.png', show_it=show_it)
        
        
