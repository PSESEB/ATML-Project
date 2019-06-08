from sklearn.naive_bayes import MultinomialNB
import pickle, sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../Sebastian'))
import BuildVectors as bv
import numpy as np



f = pickle.load( open( "dataWlabelsAndDictsSplitPen.pkl", "rb" ))

models = pickle.load( open( "MNBModels.pkl", "rb" ))

dat = f['test']

idf = pickle.load( open( "IDF.pkl", "rb" ))


idfDict = idf['idf']

wordDict = f['wordDictSmall']

BATCHSIZE = 10

labelDict = f['labelDict']


def gen(batches):
	for batch in batches:
		yield [ [bv.tfIdfTranslate(x.split(),wordDict,idfDict),bv.translate(y,labelDict)] for x,y in batch]


nbatches = len(dat)//BATCHSIZE

batchs = []
for i in range(0,nbatches):
	batch = dat[i*BATCHSIZE:(i+1)*BATCHSIZE]
	batchs += [batch]
	

if len(dat)%BATCHSIZE != 0:
	last_batch = dat[nbatches*BATCHSIZE:]
	batchs += [last_batch]



for b in gen(batchs):
	X,y = list(zip(*b))
	predL = []
	for _, v in models.items():
		if v == 'undefined':
			predL.append([0]*BATCHSIZE)
		else:
			predL.append(v.predict(X))
	preds = np.transpose(np.array(predL))
	truths = np.array(y)
	print(preds.shape,truths.shape)
	

