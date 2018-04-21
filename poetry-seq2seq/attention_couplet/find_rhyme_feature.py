import json
import numpy as np
import codecs


WORD_EMBEDDING_CONTEXT_DIM = 128
WORD_EMBEDDING_RHYME_DIM = 115
EMBEDDING_DIM = WORD_EMBEDDING_CONTEXT_DIM + WORD_EMBEDDING_RHYME_DIM
VOCAB_SIZE = 5000
RHYME_WEIGHT = 0.5
UNK = 'UNK'
SEP_TOKEN = 0
PAD_TOKEN = VOCAB_SIZE - 1
MODEL_DIR ='model'

rhyme_dict, int2ch, ch2int={}, {}, {}

with open('data/raw/rhyme_feature.json') as f:
	rhyme_dict = json.load(f)
with open('data/raw/int2ch.json') as f:
	int2ch = json.load(f)
with open('data/raw/ch2int.json') as f:
	ch2int = json.load(f)

appended_embedding=np.loadtxt('data/embedding.txt')

def find_rhyme_feature(ch):
	if ch in rhyme_dict:
		return rhyme_dict[ch]
	else:
		x = np.random.standard_normal(115)
		r = np.sqrt((x*x).sum())
		return x / r

def getEmbedding():
	word2vec=np.loadtxt('data/raw/w2v_embeddings.txt')
	embedding = np.zeros([VOCAB_SIZE, EMBEDDING_DIM])
	for i in range(VOCAB_SIZE):
		embedding[i,:WORD_EMBEDDING_CONTEXT_DIM] = word2vec[i] * (1-RHYME_WEIGHT)
		embedding[i,WORD_EMBEDDING_CONTEXT_DIM:] = np.array(find_rhyme_feature(int2ch[str(i)])) * RHYME_WEIGHT
	np.savetxt('data/embedding.txt', embedding)

def int_to_ch(i):
    i = str(i)
    return int2ch[i]


def ch_to_int(ch):
    return ch2int[ch] if ch in ch2int else ch2int[UNK]


def sentence_to_ints(sentence):
    return map(ch_to_int, sentence)


def ints_to_sentence(ints):
    return ''.join(map(int_to_ch, ints))

def process_sentence(sentence, pad_len=None, pad_token=PAD_TOKEN):
    if len(sentence) > 7:
        sentence = sentence[:7]

    sentence_ints = sentence_to_ints(sentence)

    if pad_len is not None:
        result_len = len(sentence_ints)
        for i in range(pad_len - result_len):
            sentence_ints.append(pad_token)

    return sentence_ints

def transfer_batch_predict_data(firstline):
    return np.array([process_sentence(firstline, pad_len=7)]), np.array([max(len(firstline),7)])

def gen_batch_train_data(batch_size):
    '''
    Get training data in batch major format
    '''
    data_path = 'data/duizhang_space.txt'
    maxlines = 128000
    end = False
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        line_count = 0
        batch_count = 0
        while line_count < maxlines:
            
            source, target = [],[]
            for _ in range(batch_size):
                sen = fin.readline()
                if not sen:
                    end = True
                    break
                sen = sen.strip().split(' ')

                source.append(process_sentence(sen[0], pad_len=7))
                target.append(process_sentence(sen[1], pad_len=7))
                line_count+=1   
            if end:
                break
            batch_count+=1
            if batch_count%16 == 0:
            	print "batch_count=%d"%(batch_count)

            source_padded = np.array(source)
            target_padded = np.array(target)
            source_lens = np.array([7]*batch_size)
            target_lens = np.array([7]*batch_size)

            yield source_padded, source_lens, target_padded, target_lens

