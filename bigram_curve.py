import re
import csv

from bigram_hmm import train_trigram_hmm, viterbi_trigram

corpusLines = range(1000, 41000, 1000) #[1000, 2000, 5000, 10000, 20000, 40000]
accTag = []
accSent = []

TRAIN_TAG_FILE = "data/ptb.2-21.tgs"
TRAIN_TOKEN_FILE = "data/ptb.2-21.txt"
HMM_FILE = "my_bigram_model_param.hmm"
DEVEL_TOKEN_FILE = "data/ptb.22.txt"
DEVEL_TAG_FILE = "ptb.22.out"
DEVEL_GOLD_FILE = "data/ptb.22.tgs"
CSV_OUTPUT_FILE = "accuracy.csv"

def evaluate_tag_acc(golds, hypos):
    tag_errors = 0
    sent_errors = 0
    tag_tot = 0
    sent_tot = 0

    for g, h in zip(golds, hypos):
        g = g.strip()
        h = h.strip()

        g_toks = re.split("\s+", g)
        h_toks = re.split("\s+", h)

        error_flag = False

        for i in range(len(g_toks)):
            if i >= len(h_toks) or g_toks[i] != h_toks[i]:
                tag_errors += 1
                error_flag = True

            tag_tot += 1

        if error_flag:
            sent_errors += 1

        sent_tot += 1
    
    return (tag_errors / tag_tot, sent_errors / sent_tot)

n = 0
for cLines in corpusLines:
  # Train
  train_trigram_hmm(TRAIN_TAG_FILE, TRAIN_TOKEN_FILE, HMM_FILE, cLines)
  
  # Test on development set
  viterbi_trigram(HMM_FILE, DEVEL_TOKEN_FILE, DEVEL_TAG_FILE)
  
  # Evaluate accuracy
  with open(DEVEL_GOLD_FILE) as goldFile, open(DEVEL_TAG_FILE) as hypoFile:
    golds = goldFile.readlines()
    hypos = hypoFile.readlines()
  
    if len(golds) != len(hypos):
        raise ValueError("Length is different for two files!")
  
    acct, accs = evaluate_tag_acc(golds, hypos)
    
    accTag.append(acct)
    accSent.append(accs)
  
  n += 1
  print("%d line lengths completed" % n)

# Write results to CSV  
with open(CSV_OUTPUT_FILE, mode = "w") as accuracy_file:
    acc_writer = csv.writer(accuracy_file, delimiter = ",", quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    
    for i in range(0, len(corpusLines)):
      acc_writer.writerow([corpusLines[i], accTag[i], accSent[i]])