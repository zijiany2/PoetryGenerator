from preprocess import readFileToCorpus
import collections
from collections import defaultdict
import json
inFile = "qts_tab.txt"
vocabulary = readFileToCorpus(inFile, 1)
print('Data size', len(vocabulary))
char_set = list(set(vocabulary))
print("%d characters in corpus" % (len(char_set)))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 5000

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

embed = defaultdict(list)
with open('pingshui_complete.txt') as rhyme_file:
	rhymes = [line for line in rhyme_file]

print('Number of rhymes: %d' % len(rhymes))
missing_char = []
multi_pronounce = []
for cnt, line in enumerate(rhymes):
	for ch in line:
		if ch in dictionary:
			embed[ch].extend([0]*len(rhymes))
			embed[ch][cnt] = 1

for ch in embed:
	norm = sum(embed[ch])
	if norm!= 1:
		multi_pronounce.append(ch)
		embed[ch] =  [embed[ch][i] / norm for i in range(len(rhymes))]

for ch in dictionary:
	if ch not in embed:
		missing_char.append(ch)
print("Missing Chars:")
print(missing_char)
print('Number of missing chars: %d' % len(missing_char))
print("Mutli Chars:")
print(multi_pronounce)
print('Number of multi chars: %d' % len(multi_pronounce))
with open('rhyme_feature.json','w') as jf:
	json.dump(embed, jf, indent=4)
with open('ch2int.json','w') as jf:
	json.dump(dictionary, jf, indent=4)
with open('in2ch.json','w') as jf:
	json.dump(reverse_dictionary, jf, indent=4)




