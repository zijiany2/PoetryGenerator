import json
import numpy as np
with open('rhyme_feature.json') as f:
	rhyme_dict = json.load(f)

def find_rhyme_feature(ch):
	if ch in rhyme_dict:
		return rhyme_dict[ch]
	else:
		x = np.random.standard_normal(115)
		r = np.sqrt((x*x).sum())
		return x / r