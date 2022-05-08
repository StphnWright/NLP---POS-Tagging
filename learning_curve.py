import csv
import matplotlib.pyplot as plt

CSV_ACCURACY_FILE = "accuracy.csv"
 
# Gather information
x = []
ytag = []
ysent = []

with open(CSV_ACCURACY_FILE, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",", quotechar = '"', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
      x.append(row[0])
      ytag.append(row[1])
      ysent.append(row[2])

# Plot
plt.plot(x, ytag, x, ysent)
plt.xlabel('Number of lines from the corpus')
plt.ylabel('Percent error')
plt.title('Learning Curve')
plt.legend(['Tags', 'Sentences'])
plt.show()