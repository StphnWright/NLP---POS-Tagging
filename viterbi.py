"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py hmm-file < text > tags

special keywords:
  $init_state   (an HMM state) is the single, silent start state
  $final_state  (an HMM state) is the single, silent stop state
  $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word

Usage:  python viterbi.py my.hmm < data/ptb.22.txt > my.out

"""

import sys
import re
import math

INIT_STATE = "init"
FINAL_STATE = "final"
OOV_SYMBOL = "OOV"

hmmFile = sys.argv[1]
inputFile = 'data/ptb.22.txt'

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

with open(inputFile) as inputFile:
  for line in inputFile.read().splitlines():
    w = line.split(" ");
    n = len(w);
    w.insert(0, " ")
    
    Backtrace = dict()
    
    V = {(0, INIT_STATE): 0.0}

    for i in range(1, n + 1):
      # if a word isn't in the vocabulary, rename it with the OOV symbol
      if w[i] not in Voc:
        w[i] = OOV_SYMBOL # since an OOV_SYMBOL is assigned a score during training
      
      for qq in States:
        for q in States:
          if ((qq, q) in A) and ((q, w[i]) in B) and ((i - 1, qq) in V):
            v = V[(i - 1, qq)] + A[(qq, q)] + B[(q, w[i])] # log of product
            if ((i, q) not in V) or (v > V[(i, q)]):
              # if we found a better previous state, take note!
              V[(i, q)] = v # Viterbi probability
              Backtrace[(i, q)] = qq # best previous state
              
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
      
    print(" ".join(t))
    