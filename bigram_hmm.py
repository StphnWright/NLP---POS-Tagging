"""
Implement a bigram HMM and viterbi here. 
You model should output the final tags similar to `viterbi.pl`.

Usage:  python train_trigram_hmm.py tags text > tags

"""

import re, math
from tag_acc import evalaute_tag_acc
from collections import defaultdict

def train_trigram_hmm(TAG_FILE, TOKEN_FILE, OUT_HMM_FILE, MAX_LINES = 1000000):
  
  vocab = {}
  OOV_WORD = "OOV"
  INIT_STATE = "init"
  FINAL_STATE = "final"
  
  emissions = {}
  transitions = {}
  transitionsTotal = defaultdict(int)
  emissionsTotal = defaultdict(int)
  
  lineCount = 0
  
  with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
      for tagString, tokenString in zip(tagFile, tokenFile):
  
          tags = re.split("\s+", tagString.rstrip())
          tokens = re.split("\s+", tokenString.rstrip())
          pairs = list(zip(tags, tokens))
  
          prevtag = INIT_STATE
  
          for (tag, token) in pairs:
  
              # this block is a little trick to help with out-of-vocabulary (OOV)
              # words.  the first time we see *any* word token, we pretend it
              # is an OOV.  this lets our model decide the rate at which new
              # words of each POS-type should be expected (e.g., high for nouns,
              # low for determiners).
  
              if token not in vocab:
                  vocab[token] = 1
                  token = OOV_WORD
  
              if tag not in emissions:
                  emissions[tag] = defaultdict(int)
              if prevtag not in transitions:
                  transitions[prevtag] = defaultdict(int)
  
              # increment the emission/transition observation
              emissions[tag][token] += 1
              emissionsTotal[tag] += 1
  
              transitions[prevtag][tag] += 1
              transitionsTotal[prevtag] += 1
  
              prevtag = tag
              
          # don't forget the stop probability for each sentence
          if prevtag not in transitions:
              transitions[prevtag] = defaultdict(int)
  
          transitions[prevtag][FINAL_STATE] += 1
          transitionsTotal[prevtag] += 1
          
          # increment the count by 1
          lineCount += 1
          
          # Check if limit reached
          if (lineCount > MAX_LINES):
            break
  
  with open(OUT_HMM_FILE, 'w') as hmmFile:
    for prevtag in transitions:
        for tag in transitions[prevtag]:
            hmmFile.write(("trans %s %s %s\n" % (prevtag, tag, float(transitions[prevtag][tag]) / transitionsTotal[prevtag])))
    
    for tag in emissions:
        for token in emissions[tag]:
            hmmFile.write(("emit %s %s %s \n" % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])))


def viterbi_trigram(hmmFile, inputFile, outFile, isJV = False):

  INIT_STATE = "init"
  FINAL_STATE = "final"
  OOV_SYMBOL = "OOV"
  
  States = dict()
  A = dict()
  B = dict()
  Voc = dict()
  
  # read in the HMM and store the probabilities as log probabilities
  regexTrans = "trans\s+(\S+)\s+(\S+)\s+(\S+)"
  regexEmit = "emit\s+(\S+)\s+(\S+)\s+(\S+)" 
  
  with open(hmmFile) as hmmFile:
    for line in hmmFile.read().splitlines():
      
      matchEmit = re.match(regexEmit, line)
      if matchEmit:
        q, w, p = matchEmit.groups()
        B[(q, w)] = math.log(float(p))
        States[q] = 1
        Voc[w] = 1
      
      else:
        matchTrans = re.match(regexTrans, line)
        if matchTrans:
          qq, q, p = matchTrans.groups()
          A[(qq, q)] = math.log(float(p))
          States[qq] = 1
          States[q] = 1      
  
  with open(inputFile) as inputFile, open(outFile, 'w') as outFile:
    for line in inputFile.read().splitlines():
      if isJV:
        w = line.split();
      else:
        w = line.split(" ");
      n = len(w);
      w.insert(0, "")
      
      Backtrace = dict()
      
      V = {(0, INIT_STATE): 0.0}
  
      for i in range(1, n + 1):
        # if a word isn't in the vocabulary, rename it with the OOV symbol
        if w[i] not in Voc:
          w[i] = OOV_SYMBOL  # since an OOV_SYMBOL is assigned a score during training
        
        for qq in States:
          for q in States:
            if ((qq, q) in A) and ((q, w[i]) in B) and ((i - 1, qq) in V):
              v = V[(i - 1, qq)] + A[(qq, q)] + B[(q, w[i])]  # log of product
              if ((i, q) not in V) or (v > V[(i, q)]):
                # if we found a better previous state, take note!
                V[(i, q)] = v  # Viterbi probability
                Backtrace[(i, q)] = qq  # best previous state
                
      # this handles the last of the Viterbi equations, the one that brings
      # in the final state.
      goal = float("-inf")
      foundgoal = False
      q = INIT_STATE
  
      for qq in States:
        if ((qq, FINAL_STATE) in A) and ((n, qq) in V):
          v = V[(n, qq)] + A[(qq, FINAL_STATE)]
          if (not foundgoal) or (v > goal):
            # we found a better path; remember it
            goal = v
            foundgoal = True
            q = qq
      
      # this is the backtracking step.
      t = []
      if foundgoal:
        for i in range(n, 1, -1):
          t.append(Backtrace[i, q])
          q = Backtrace[i, q]
        t.reverse()
        
        if isJV:
          t.append(".")
        
      outFile.write(' '.join(t) + "\n")

if __name__ == "__main__":

  TRAIN_TAG_FILE = "data/ptb.2-21.tgs"
  TRAIN_TOKEN_FILE = "data/ptb.2-21.txt"
  HMM_FILE = "data/my_bigram_model.hmm"
  DEVEL_TOKEN_FILE = "data/ptb.22.txt"
  DEVEL_TAG_FILE = "data/bigram_ptb.22.out"
  DEVEL_GOLD_FILE = "data/ptb.22.tgs"
  TEST_TOKEN_FILE = "data/ptb.23.txt"
  TEST_TAG_FILE = "data/bigram_ptb.23.out"
  
  # Train
  train_trigram_hmm(TRAIN_TAG_FILE, TRAIN_TOKEN_FILE, HMM_FILE)
  
  # Test on development set
  viterbi_trigram(HMM_FILE, DEVEL_TOKEN_FILE, DEVEL_TAG_FILE)
  
  # Evaluate accuracy
  with open(DEVEL_GOLD_FILE) as goldFile, open(DEVEL_TAG_FILE) as hypoFile:
    golds = goldFile.readlines()
    hypos = hypoFile.readlines()

    if len(golds) != len(hypos):
        raise ValueError("Length is different for two files!")

    evalaute_tag_acc(golds, hypos)
  
  # Test on test set
  viterbi_trigram(HMM_FILE, TEST_TOKEN_FILE, TEST_TAG_FILE)
  
