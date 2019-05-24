import pickle
import math

labelFile = open('id2class_eurlex_eurovoc.qrels','r')

labels = {}
i = 0
labelDict = {}
for x in labelFile:
	temp = x.split()

	if temp[1] in labels:
		labels[temp[1]].append(temp[0])
	else:
		labels[temp[1]] = [temp[0]]

	if temp[0] not in labelDict:
		labelDict[temp[0]] = i
		i = i+1




data =  pickle.load( open( "dictsAndData.pkl", "rb" ))

docs = data['RawData']

biDict = data['BiDict']





properBiDict = {}
i = 0
for x in biDict:
	if x[0] not in properBiDict:
		properBiDict[x[0]] = i
		i +=1


wDFull = data['worddict']
properFullDict = {}

i=0
for x in wDFull:
	if x not in properFullDict:
		properFullDict[x] = i
		i +=1

wDSmall = data['worddictS']
properSmallDict = {}
i = 0
for x in wDSmall:
	if x[0] not in properSmallDict:
		properSmallDict[x[0]] = i
		i +=1





DataPoints = []
for doc in docs:
	id_d = str(int(doc[0]))
	if id_d in labels:
		datapoint = [doc[1],labels[id_d]]
		DataPoints.append(datapoint)
	else:
		print("ID ",id_d," has no LABELS")



###Calc IDF
idf = {}
for x in wDFull:
	idf[x] = math.log(len(DataPoints)/wDFull[x])


saveFile = {}

saveFile['data'] = DataPoints

saveFile['labelDict'] = labelDict
saveFile['biDict'] = properBiDict
saveFile['wordDictFull'] = properFullDict
saveFile['wordDictSmall'] = properSmallDict
saveFile['freqcountWords'] = wDFull
saveFile['idf'] = idf

pickle_out = open("dataWlabelsAndDicts.pkl","wb")
pickle.dump(saveFile, pickle_out)
pickle_out.close()