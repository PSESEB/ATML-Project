import pickle
import numpy as np

def translate(inp, dic, invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	vec = np.zeros(len(dic))
	for x in inp:
		if x in dic:
			vec[dic[x]] += 1
	return vec


def translateNGram(inp,dic, n,invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	vec = np.zeros(len(dic))
	for i in range(0,len(inp)-n+1):
		ngram = ""
		for x in range(0,n):
			ngram += inp[i+x]+" "
		ngram = ngram.rstrip()
		if ngram in dic:
			vec[dic[ngram]] += 1
	return vec



def tfIdfTranslate(inp, dic, idf, invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	tf = {}
	for word in inp:
		if word in dic:
			if word in tf:
				tf[word] += 1
			else:
				tf[word] = 1
	vec = np.zeros(len(dic))

	for t in tf:
		vec[dic[t]] = tf[t]/idf[t]
	return vec

doc = ["this", "is", "a", "document", "is","a","is"]
dicti = {"this is":1, "is a": 0, "a document":3, "document is":2}

print(translateNGram(doc,dicti,2))


data =  pickle.load( open( "dataWlabelsAndDicts.pkl", "rb" ))
points = data['idf']

print(points)


