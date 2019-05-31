import tensorflow as tf
import pickle, sys, os


sys.path.insert(1, os.path.join(sys.path[0], '../Sebastian'))
import BuildVectors as bv
import random

f = pickle.load( open( "dataWlabelsAndDicts.pkl", "rb" ))


BATCHSIZE = 1000
NN_BATCHSIZE = 64

nbatches = len(f['data'])//BATCHSIZE


data = f['data']
random.shuffle(data)
wdfull = f['wordDictFull']
labelDict = f['labelDict']

batches = []
for i in range(0,nbatches):
	batch = data[i*BATCHSIZE:(i+1)*BATCHSIZE]
	batches += [batch]
	

last_batch = data[nbatches*BATCHSIZE:]
batches += [last_batch]


def gen():
	for batch in batches:
		for b in batch:
			yield (bv.sequenceTranslate(b[0].split(),wdfull),bv.translate(b[1],labelDict))


ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([None]), tf.TensorShape([None])))





ds = ds.shuffle(20000).repeat().padded_batch(NN_BATCHSIZE,padded_shapes=([None],[None]), padding_values=(tf.constant(-1, dtype=tf.int64)
                                                 ,tf.constant(-1, dtype=tf.int64)))

iterator = ds.make_one_shot_iterator()
next_batch = iterator.get_next()

# try it out
with tf.Session() as sess:
    i = 0
    try:
        while i < 1000:
            txt,lbl = sess.run(next_batch)

            lengths = tf.math.count_nonzero(txt+1,1)

            lengths_transposed = tf.expand_dims(lengths, 1)

            ranges = tf.range(0,txt.shape[1],1,dtype=tf.int64)

            range_row = tf.expand_dims(ranges, 0)
            mask = tf.cast(tf.less(range_row, lengths_transposed), tf.int64)

            print(txt.shape,lbl.shape)
            print(i)
            i += 1
            # we could e.g. print the shapes of the outputs to make sure they
            # make sense (should be (784,), ())
    except tf.errors.OutOfRangeError:
        print("Done!")

