from algorithms.svea import SVEA
from algorithms.sac import SAC
algorithm = {
	'svea': SVEA,
	'sac': SAC
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
