import logging
import gzip
import csv

sts_dataset_path = '../Classification/data/stsbenchmark.tsv.gz'

f=open('STS_test.csv', 'w',encoding='utf-8', newline='')
wr = csv.writer(f)


STS_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            wr.writerow([str(row['sentence1']), str(row['sentence2']), str(score)])


f.close()

