from region import Region, make_regions
from person import Person, PersonType
from distortion import distortion

import matplotlib.pyplot as plt
import matplotlib.colors
from collections import defaultdict
import numpy as np

def place_voters_ax(voters,ax=None, opacity=0.1,show_names=False):
    if ax == None:
        fig,ax = plt.subplots()
    scatters = {}
    for v in voters:
        color_key = tuple(v.color) if isinstance(v.color, np.ndarray) else v.color
        the_guy = scatters.get((color_key,v.marker),None)
        if the_guy is None:
            the_guy = np.array([[],[]])
        scatters[(color_key,v.marker)] = np.append(the_guy, [[v.pos[0]],[v.pos[1]]], axis=1)
        if show_names:
            ax.annotate(v.name, v.pos)
    for col_mark, arr in scatters.items():
        ax.scatter(arr[0,:],arr[1,:], color=col_mark[0], marker=col_mark[1],alpha=opacity)
    return ax

def add_dists_ax(ax,total_dist, group_dists, group_colors, x=0.95, y=0.95,offset=0.05,fontsize=10):
    ax.text(x, y, f'Distortion: {round(total_dist,3)}',
    horizontalalignment='right',
    verticalalignment='top',
    transform=ax.transAxes, # Use axes coordinates (0 to 1)
    fontsize=10,
    #bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5) # Optional: add a text box
    )

    i = 1
    siz = offset
    for name,group_dist in group_dists.items():
        ax.text(x, y-i*offset, f'Distortion: {round(group_dist,3)}',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes, # Use axes coordinates (0 to 1)
        fontsize=fontsize,
        color=group_colors[name]
        #bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5) # Optional: add a text box
        )
        i+=1
    return ax

def make_group_colors(group_dists,voters):
    group_colors = {}
    for key in group_dists.keys():
        for v in voters:
            if v.name == key:
                group_colors[v.name] = v.color
                break
    return group_colors

def build_ax(result, ax, display_cands=True, show_cand_names=True,show_distortion=True, show_group_dist=True, opacity=0.1,x_dist=0.95,y_dist=0.95,winner_color='m',cand_color='g'):
    cands_for_disp = []
    candidates = result.get('candidates',[])
    winners = result.get('winners',[])
    voters = result.get('voters',[])
    group_dists = result.get('group_dists', {})
    for k,pos in candidates.items(): 
        color = cand_color
        if k in winners:
            color = winner_color
        cand_disp = Person(pos,color=color,marker='+',name=k)
        cands_for_disp.append(cand_disp)
    ax = place_voters_ax(voters,opacity=opacity,ax=ax,show_names=False)
    if display_cands:
        ax = place_voters_ax(cands_for_disp,opacity=None,ax=ax,show_names=show_cand_names)
    if show_distortion:
        if show_group_dist:
            group_colors = make_group_colors(group_dists, voters)
            ax = add_dists_ax(ax,result['distortion'],group_dists,group_colors,x=x_dist,y=y_dist)
        else:
            ax = add_dists_ax(ax,result['distortion'],{},{},x=x_dist,y=y_dist)
    return ax

def display(result, ax_kwargs=None, opacity=0.1,xlim=(-1,1),ylim=(1,1), show_it=True,save_at=None):
    if ax_kwargs == None:
        ax_kwargs = {}
    fig,ax = plt.subplots()
    ax = build_ax(result,ax,**ax_kwargs)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if save_at != None:
        plt.savefig(save_at, bbox_inches='tight', dpi=300)
    if show_it:
        plt.show()
    return ax

def prepare_candidates_for_display(cand_pos, cand_colors, winners=None, winner_color = 'm',winner_marker='+',cand_markers=None):
    cands = []
    for k,pos in cand_pos.items():
        color = cand_colors[k]
        marker = '+'
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
