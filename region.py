from person import Person, PersonType, circle_gauss_system
import person
import numpy as np

class Region:
    def __init__(self,voters,name="",subregions=[],deterministic=False):
        self.voters=voters
        self.voter_types=list(voters.keys())
        self.amounts = list(voters.values())
        self.name = name
        self.subregions = subregions
        assert len(self.voters) == len(self.amounts), "Dude What"
        
        tot = sum(self.amounts)
        self.population = tot
        self.proportions = [x/tot for x in self.amounts]
        self.deterministic = deterministic

    def gen_one_random(self):
        my_guy = np.random.choice(self.voter_types, p=self.proportions)
        return my_guy()

    def gen_random(self, num_samples=1):
        if self.deterministic:
            arr = np.array([])
            for v_type, amt in self.voters.items():
                add_arr = np.array([v_type() for i in range(0,amt)])
                arr = np.concatenate((arr,add_arr))
            return arr
        else:
            return np.array([self.gen_one_random() for i in range(0,num_samples)])

    def combine(regions,name=None,deterministic=False):
        if name == None:
            name = "__".join(region.name for region in regions)
        combined = {}
        for region in regions:
            for voter_type, amount in region.voters.items():
                new_voter_type = voter_type.copy()
                new_voter_type.region = region.name
                #print(new_voter_type.region)
                combined[new_voter_type] = amount
                #combined[voter_type] = combined.get(voter_type, 0) + amount
        return Region(combined,name,
                      subregions=regions,deterministic=deterministic)
        
def make_regions():
    regions = {}

    def bloc(mean, cov, name, region):
        return PersonType.correlated_gaussian(mean=mean, cov=cov, name=name, region=region)

    cov_tight = [[0.20, 0.05], [0.05, 0.20]]
    cov_wide  = [[0.45, 0.10], [0.10, 0.45]]

    # NORTH
    north = "North"
    regions[north] = Region(
        {
            bloc([ 0.2,  0.6], cov_tight, "Urban", north): 0.30,
            bloc([-0.1,  0.7], cov_wide,  "Rural", north): 0.45,
            bloc([ 0.0,  0.2], cov_tight, "SmallTown", north): 0.25,
        },
        name=north,
    )

    # SOUTH
    south = "South"
    regions[south] = Region(
        {
            bloc([-0.1, -0.2], cov_tight, "Urban", south): 0.35,
            bloc([-0.3, -0.4], cov_wide,  "Rural", south): 0.40,
            bloc([ 0.2, -0.1], cov_tight, "Coastal", south): 0.25,
        },
        name=south,
    )

    # EAST
    east = "East"
    regions[east] = Region(
        {
            bloc([-0.4,  0.1], cov_wide,  "Rural", east): 0.55,
            bloc([-0.1,  0.0], cov_tight, "Urban", east): 0.25,
            bloc([-0.2,  0.4], cov_tight, "Industrial", east): 0.20,
        },
        name=east,
    )

    # WEST
    west = "West"
    regions[west] = Region(
        {
            bloc([ 0.5,  0.2], cov_tight, "Urban", west): 0.40,
            bloc([ 0.2,  0.1], cov_wide,  "Rural", west): 0.35,
            bloc([ 0.6, -0.1], cov_tight, "Coastal", west): 0.25,
        },
        name=west,
    )

    # CENTRAL
    central = "Central"
    regions[central] = Region(
        {
            bloc([ 0.0,  0.5], cov_wide,  "Rural", central): 0.55,
            bloc([ 0.2,  0.3], cov_tight, "SmallTown", central): 0.25,
            bloc([ 0.3,  0.6], cov_tight, "Urban", central): 0.20,
        },
        name=central,
    )

    # NORTHEAST (smaller, more “regional-party” feeling)
    ne = "Northeast"
    regions[ne] = Region(
        {
            bloc([-0.5, -0.1], cov_tight, "Hill", ne): 0.45,
            bloc([-0.3,  0.0], cov_tight, "Town", ne): 0.35,
            bloc([-0.2,  0.3], cov_wide,  "Rural", ne): 0.20,
        },
        name=ne,
    )

    return regions

def tri_party(sigma=0.5):
    regions = {}
    parties = person.circle_gauss_system(3,sigma=sigma,size=0.6)
    north = "North"
    regions[north] = Region(
        {
            parties[0]: 700,
            parties[1]: 200,
            parties[2]: 100
        }
    )

    central = "Central"
    regions[central] = Region(
        {
            parties[0]: 150,
            parties[1]: 800,
            parties[2]: 50 
        }
    )

    south = "South"
    regions[south] = Region(
        {
            parties[0]: 350,
            parties[1]: 50,
            parties[2]: 600
        }
    )
    return regions

def two_bloc_weighted(bias=0.5, sigma=0.5, size=0.6,tot=1000,parties=None):
    regions = {}
    if parties == None:
        parties = person.circle_gauss_system(2,sigma=sigma,size=size,offset=np.pi/2)
    state_1 = "Party1State"
    state_2 = "Party2State"
    each_state = tot//2
    regions[state_1] = Region({
        parties[0]: int(np.floor(each_state*bias)),
        parties[1]: each_state-int(np.floor(each_state*bias))
    })

    regions[state_2] = Region({
        parties[0]: each_state-int(np.floor(each_state*bias)),
        parties[1]: int(np.floor(each_state*(bias)))
    })

    return regions
