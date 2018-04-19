import numpy as np

from find_rhyme_feature import rhyme_dict, int2ch, ch2int, WORD_EMBEDDING_CONTEXT_DIM, WORD_EMBEDDING_RHYME_DIM,appended_embedding

def get_rhyme(ch):
	ch = ch.decode('utf-8')
	idx = 0
	if ch in rhyme_dict:
		idx = np.argmax(rhyme_dict[ch])
	return idx


def rhyme_boosting(ch, rhyme):
	ch = ch.decode('utf-8')
	max_score = 0
	max_ch = ch
	for candidate in rhyme_dict:
		if rhyme_dict[candidate][rhyme] >1e-5:
			temp_score = compute_embedding_socre(ch, candidate) 
			if temp_score > max_score: 
				max_score = temp_score
				max_ch = candidate
	return max_ch.encode('utf-8')


def compute_embedding_socre(a, b):
	ia, ib = ch2int[a], ch2int[b]
	return np.dot(appended_embedding[ia], appended_embedding[ib])
