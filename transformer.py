import logging
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

SEQ_LEN = 50
NUM_EPOCHS = 1 # default of 20
NUM_STEPS = 0
ALPHA_GD = 0.0001 # default 0.01
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

"""
TRAIN_TAG_FILE = "data/ptb.2-21.tgs"
TRAIN_TOKEN_FILE = "data/ptb.2-21.txt"
DEVEL_TOKEN_FILE = "data/ptb.22.txt"
DEVEL_TAG_FILE = "bigram_ptb.22.out"
DEVEL_GOLD_FILE = "data/ptb.22.tgs"
"""

TRAIN_TAG_FILE = "data/btb.train.tgs"
TRAIN_TOKEN_FILE = "data/btb.train.txt"
DEVEL_TOKEN_FILE = "data/btb.test.txt"
DEVEL_TAG_FILE = "data/btb.btb_bigram_test.out"
DEVEL_GOLD_FILE = "data/btb.test.tgs"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# initialize model and tokenizer
bert = TFAutoModel.from_pretrained("distilbert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

# read data and drop dupes

#arr = df['Sentiment'].values  # take sentiment column in df as array
#labels = np.zeros((arr.size, arr.max()+1))  # initialize empty (all zero) label array
#labels[np.arange(arr.size), arr] = 1  # add ones in indices where we have a value

# define function to handle tokenization
def tokenize(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

# Read data
#x = []
#y = []
'''
with open(TRAIN_TAG_FILE) as tagFile, open(TRAIN_TOKEN_FILE) as tokenFile:
    for tagString, tokenString in zip(tagFile, tokenFile):
        Xids[i, :], Xmask[i, :] = tokenize(sentence)
        i += 1
        #tags = re.split("\s+", tagString.rstrip())
        #tokens = re.split("\s+", tokenString.rstrip())
        #pairs = list(zip(tags, tokens))
        
        #for (tag, token) in pairs:
        #  x.append(tag)
        #  y.append(token)


# Count unique tag types
tagsAll = []
with open(TRAIN_TAG_FILE) as tagFile:
  for tagString in tagFile:
      tags = re.split("\s+", tagString.rstrip())
      
      for tag in tags:
          tagsAll.append(tag)
      #  x.append(tag)
      #  y.append(token)

# taking an input list
tagsUnique = []
TAG_COUNT = 0
 
# traversing the array
for item in tagsAll:
    if item not in tagsUnique:
        TAG_COUNT += 1
        tagsUnique.append(item)
'''
  
def convert_data_to_tf(TOKEN_FILE, TAG_FILE, SEQUENCE_LENGTH = SEQ_LEN):
  # Count sentences
  ns = 0
  with open(TRAIN_TOKEN_FILE) as tokenFile:
      for tokenString in tokenFile:
          ns += 1
  
  Xids = np.zeros((ns, SEQUENCE_LENGTH))
  Xmask = np.zeros((ns, SEQUENCE_LENGTH))
  labels = np.zeros((ns, SEQUENCE_LENGTH))
  Xmasklabels = np.zeros((ns, SEQUENCE_LENGTH))
  
  i = 0
  with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
      for tagString, tokenString in zip(tagFile, tokenFile):
          Xids[i, :], Xmask[i, :] = tokenize(tokenString)
          labels[i, :], Xmasklabels[i, :] = tokenize(tagString)
          # Xids[i, :] = i
          # Xmask[i, :] = 1
          # labels[i, :] = i
          i += 1
  
  # create tensorflow dataset object
  dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
  
  # restructure dataset format for BERT
  def map_func(input_ids, masks, labels):
      return {'input_ids': input_ids, 'attention_mask': masks}, labels
    
  dataset = dataset.map(map_func)  # apply the mapping function
  
  # shuffle and batch the dataset
  dataset = dataset.shuffle(10000).batch(32)
  
  return dataset


dsTrain = convert_data_to_tf(TRAIN_TOKEN_FILE, TRAIN_TAG_FILE)

# build the model
input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]  # we only keep tensor 0 (last_hidden_state)

X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
y = tf.keras.layers.Dense(SEQ_LEN, activation='softmax', name='outputs')(X)  # adjust based on number of sentiment classes

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

# freeze the BERT layer
model.layers[2].trainable = False

# compile the model
optimizer = tf.keras.optimizers.Adam(ALPHA_GD)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

# and train it
if (NUM_STEPS > 0):
  history = model.fit(dsTrain, epochs=NUM_EPOCHS, steps_per_epoch=NUM_STEPS)
else:
  history = model.fit(dsTrain, epochs=NUM_EPOCHS)


# Test the model
dsTest = convert_data_to_tf(DEVEL_TOKEN_FILE, DEVEL_GOLD_FILE)
'''
if (NUM_STEPS > 0):
  results = model.evaluate(dsTest, steps=NUM_STEPS)
else:
  results = model.evaluate(dsTest)
print("test loss, test acc:", results)

if (NUM_STEPS > 0):
  prediction = model.predict(dsTest, steps=NUM_STEPS)
else:
  prediction = model.predict(dsTest)

for pred in prediction:
  dec = tokenizer.decode(pred)
  print(dec)
'''