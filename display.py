from region import Region, make_regions
from person import Person, PersonType
from distortion import distortion

import matplotlib.pyplot as plt
import matplotlib.colors
from collections import defaultdict

def prepare_candidates_for_display(cand_pos, cand_colors, winners=None, winner_color = 'm',winner_marker='^',cand_markers=None):
    cands = []
    for k,pos in cand_pos.items():
        color = cand_colors[k]
        marker = 'o'
        if winners != None and k in winners:
            color = winner_color
            marker = winner_marker
        cands.append((k,pos,color,marker))
    return cands

def gen_voter_display_type(voter_names, voter_regions):
    marker_list = ['o','x','v','^','1','s','P','+','*','2']
    shape_type = {}
    color_type = {}
    tot_voters = len(voter_names)
    assert len(voter_regions) <= len(marker_list)
    voter_name_ct = 0
    for v in voter_names:
        region_ct = 0
        voter_color = tuple(matplotlib.colors.hsv_to_rgb((voter_name_ct/tot_voters,1,1)))
        for r in voter_regions:
            shape_type[r+'.'+v] = marker_list[region_ct]
            color_type[r+'.'+v] = voter_color
            region_ct += 1
        voter_name_ct += 1
    return shape_type, color_type

def display_by_type(voter_data, cands, shape_type, color_type,ax=None,opacity=0.5):
    if ax == None:
        fig,ax = plt.subplots()
    data_sorted = defaultdict(list)
    
    for d in voter_data:
        s = shape_type[d.identifier]
        c = color_type[d.identifier]
        data_sorted[(s,c)].append(d.pos)
    for k,v in data_sorted.items():
        ax.scatter([x[0] for x in v],[x[1] for x in v],marker=k[0],color=k[1],alpha=opacity)
    for name,pos,color,marker in cands:
        ax.scatter(pos[0],pos[1], marker=marker,color=color)
        ax.annotate(name,pos)
    
    return ax
