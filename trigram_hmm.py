"""
Implement a trigrm HMM and viterbi here. 
You model should output the final tags similar to `viterbi.pl`.

Usage:  python train_trigram_hmm.py tags text > tags

"""

import re, math
from collections import defaultdict 
import numpy as np
from tag_acc import evalaute_tag_acc

OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"


def train_trigram_hmm(TAG_FILE, TOKEN_FILE, OUT_HMM_FILE="model_trigram_interp.hmm", MAX_LINES=1000000):
    
  vocab = {}
  
  emissions = {}
  transitions = {}
  transitionsTotal = defaultdict(int)
  emissionsTotal = defaultdict(int)
  totalN = 0
  lineCount = 0
  
  with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
      for tagString, tokenString in zip(tagFile, tokenFile):
  
          tags = re.split("\s+", tagString.rstrip())
          tokens = re.split("\s+", tokenString.rstrip())
          pairs = list(zip(tags, tokens))
  
          prevprevtag = INIT_STATE
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
              if prevprevtag not in transitions:
                transitions[prevprevtag] = defaultdict(dict)
              if prevtag not in transitions[prevprevtag]:
                transitions[prevprevtag][prevtag] = defaultdict(int)
              
              if prevprevtag not in transitionsTotal:
                transitionsTotal[prevprevtag] = defaultdict(int)
                
              # increment the emission/transition observation
              emissions[tag][token] += 1
              emissionsTotal[tag] += 1
              totalN += 1
  
              transitions[prevprevtag][prevtag][tag] += 1
              transitionsTotal[prevprevtag][prevtag] += 1
              
              prevprevtag = prevtag
              prevtag = tag
  
          # don't forget the stop probability for each sentence
          if prevprevtag not in transitions:
            transitions[prevprevtag] = defaultdict(int)
          if prevtag not in transitions[prevprevtag]:
            transitions[prevprevtag][prevtag] = defaultdict(int)
  
          transitions[prevprevtag][prevtag][FINAL_STATE] += 1
          
          if prevprevtag not in transitionsTotal:
            transitionsTotal[prevprevtag] = defaultdict(int)
          
          transitionsTotal[prevprevtag][prevtag] += 1
          
          prevprevtag = prevtag
          prevtag = tag
          
          if prevprevtag not in transitions:
            transitions[prevprevtag] = defaultdict(dict)
          if prevprevtag not in transitionsTotal:
            transitionsTotal[prevprevtag] = defaultdict(int)
          
          transitions[prevprevtag][FINAL_STATE] = defaultdict(int)
          transitions[prevprevtag][FINAL_STATE][FINAL_STATE] += 1
          transitionsTotal[prevprevtag][FINAL_STATE] += 1
          
          # increment the count by 1
          lineCount += 1
          
          # Check if limit reached
          if (lineCount > MAX_LINES):
            break
  
  # Initialize lambdas
  L = [0, 0, 0]
  t = [0, 0, 0]
  c = [0, 0, 0, 0]

  with open(OUT_HMM_FILE, 'w') as hmmFile: 
    # Write the total number of words
    c[0] = totalN
    hmmFile.write("countAllWords %s\n" % c[0]) 
    
    # Write the unigrams and emissions
    for tag in emissions:
        hmmFile.write("countUnigram %s %s\n" % (tag, emissionsTotal[tag]))
        for token in emissions[tag]:
            hmmFile.write(("emit %s %s %s \n" % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])))
    
    # Write the bigrams and trigrams
    for prevprevtag in transitions:
      for prevtag in transitions[prevprevtag]:
        hmmFile.write("countBigram %s %s %s\n" % (prevprevtag, prevtag, transitionsTotal[prevprevtag][prevtag]))
        for tag in transitions[prevprevtag][prevtag]:
          c[3] = transitions[prevprevtag][prevtag][tag]
          hmmFile.write("countTrigram %s %s %s %s\n" % (prevprevtag, prevtag, tag, c[3]))
          
          # Count unigram and bigram
          c[1] = emissionsTotal[tag]
          
          if (prevtag in transitionsTotal) and (tag in transitionsTotal[prevtag]):
            c[2] = transitionsTotal[prevtag][tag]
          else:
            c[2] = 0  
          
          # Decide which lambda to increment
          for i in range(0, 3):
            try:
              t[i] = float(c[i + 1] - 1) / (c[i] - 1)
            except ZeroDivisionError:
              t[i] = 0
          
          L[np.argmax(t)] += c[3]

    # Normalize lambdas
    totalL = sum(L)
    L_norm = [x / totalL for x in L]
    hmmFile.write("lambdaValues %s %s %s\n" % (L_norm[0], L_norm[1], L_norm[2])) 


def viterbi_trigram(inputFile, outFile, hmmFilename="model_trigram_interp.hmm", isJV=False):
  
  States = dict()
  A = dict()
  B = dict()
  Voc = dict()
  
  countUnigram = defaultdict(int)
  countBigram = defaultdict(int)
  
  # read in the HMM and store the probabilities as log probabilities
  regexEmit = "emit\s+(\S+)\s+(\S+)\s+(\S+)" 
  regexLambda = "lambdaValues\s+(\S+)\s+(\S+)\s+(\S+)"
  regexAllWords = "countAllWords\s+(\S+)"
  regexUnigram = "countUnigram\s+(\S+)\s+(\S+)"
  regexBigram = "countBigram\s+(\S+)\s+(\S+)\s+(\S+)"
  regexTrigram = "countTrigram\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
  
  # Create emits and read everything except trigrams   
  with open(hmmFilename) as hmmFile:
    for line in hmmFile.read().splitlines():
      
      match = re.match(regexEmit, line)
      if match:
        q, w, p = match.groups()
        B[(q, w)] = math.log(float(p))
        States[q] = 1
        Voc[w] = 1
        continue
      
      match = re.match(regexUnigram, line)
      if match:
        tag, c = match.groups()
        countUnigram[tag] = float(c)
        continue
      
      match = re.match(regexBigram, line)
      if match:
        prevtag, tag, c = match.groups()
        if prevtag not in countBigram:
          countBigram[prevtag] = defaultdict(int)
        countBigram[prevtag][tag] = float(c)
        continue
      
      match = re.match(regexLambda, line)
      if match:
        lambdaUniStr, lambdaBiStr, lambdaTriStr = match.groups()
        lambdaUni = float(lambdaUniStr)
        lambdaBi = float(lambdaBiStr)
        lambdaTri = float(lambdaTriStr)
        continue
      
      match = re.match(regexAllWords, line)
      if match:
        totalNStr = match.groups()
        totalN = float(totalNStr[0])
  
  with open(hmmFilename) as hmmFile:
    for line in hmmFile.read().splitlines(): 
        match = re.match(regexTrigram, line)
        if match:
          qqq, qq, q, countTrigramStr = match.groups()
          countTrigram = float(countTrigramStr)

          p = 0
          if (qqq in countBigram) and (qq in countBigram[qqq]) and (countBigram[qqq][qq] > 0):
            p += lambdaTri * countTrigram / countBigram[qqq][qq]
          if (qq in countBigram) and (q in countBigram[qq]) and (qq in countUnigram) and (countUnigram[qq] > 0):
            p += lambdaBi * countBigram[qq][q] / countUnigram[qq]
          if (q in countUnigram):
            p += lambdaUni * countUnigram[q] / totalN
          
          A[(qqq, qq, q)] = math.log(float(p))
          States[qqq] = 1
          States[qq] = 1
          States[q] = 1
              
  with open(inputFile) as inputFile, open(outFile, "w") as outFile:
    for line in inputFile.read().splitlines():
      if isJV:
        w = line.split();
      else:
        w = line.split(" ");
      n = len(w);
      w.insert(0, " ")
      
      Backtrace = dict()
      
      V = {(0, INIT_STATE, INIT_STATE): 0.0}
  
      for i in range(1, n + 1):
        # if a word isn't in the vocabulary, rename it with the OOV symbol
        if w[i] not in Voc:
          w[i] = OOV_WORD  # since an OOV_WORD is assigned a score during training
        
        for qqq in States:
          for qq in States:
            for q in States:
              if ((qqq, qq, q) in A) and ((q, w[i]) in B) and ((i - 1, qqq, qq) in V):
                v = V[(i - 1, qqq, qq)] + A[(qqq, qq, q)] + B[(q, w[i])]  # log of product
                if ((i, qq, q) not in V) or (v > V[(i, qq, q)]):
                  # if we found a better previous state, take note!
                  V[(i, qq, q)] = v  # Viterbi probability
                  Backtrace[(i, qq, q)] = qqq  # best previous state
                
      # this handles the last of the Viterbi equations, the one that brings
      # in the final state.
      goal = float("-inf")
      foundgoal = False
      q = INIT_STATE
      qq = INIT_STATE
      
      for qqcurr in States:
        for qcurr in States:
          if ((qqcurr, qcurr, FINAL_STATE) in A) and ((n, qqcurr, qcurr) in V):
            v = V[(n, qqcurr, qcurr)] + A[(qqcurr, qcurr, FINAL_STATE)]
            if (not foundgoal) or (v > goal):
              # we found a better path; remember it
              goal = v
              foundgoal = True
              q = qcurr
              qq = qqcurr
      
      # this is the backtracking step.
      t = []
      if foundgoal:
        t = [qq]
        for i in range(n, 2, -1):
          t.append(Backtrace[i, qq, q])
          qqtemp = qq
          qq = Backtrace[i, qq, q]
          q = qqtemp
        t.reverse()
        
        if isJV:
          t.append(".")
      else:
        t.append("")
      
      if t:
          outFile.write(" ".join(t) + "\n")

          
if __name__ == "__main__":

  TRAIN_TAG_FILE = "data/ptb.2-21.tgs"
  TRAIN_TOKEN_FILE = "data/ptb.2-21.txt"
  HMM_FILE = "data/my_trigram_model.hmm"
  DEVEL_TOKEN_FILE = "data/ptb.22.txt"
  DEVEL_TAG_FILE = "data/ptb.22.out"
  DEVEL_GOLD_FILE = "data/ptb.22.tgs"
  TEST_TOKEN_FILE = "data/ptb.23.txt"
  TEST_TAG_FILE = "data/ptb.23.out"

  # Train
  train_trigram_hmm(TRAIN_TAG_FILE, TRAIN_TOKEN_FILE, HMM_FILE)

  # Test on development set
  viterbi_trigram(DEVEL_TOKEN_FILE, DEVEL_TAG_FILE, HMM_FILE)
  
  # Evaluate accuracy
  with open(DEVEL_GOLD_FILE) as goldFile, open(DEVEL_TAG_FILE) as hypoFile:
    golds = goldFile.readlines()
    hypos = hypoFile.readlines()

    if len(golds) != len(hypos):
        raise ValueError("Length is different for two files!")

    evalaute_tag_acc(golds, hypos)
  
  # Test on test set
  viterbi_trigram(HMM_FILE, TEST_TOKEN_FILE, TEST_TAG_FILE)
