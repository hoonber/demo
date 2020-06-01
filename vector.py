from bert_serving.client import BertClient
import numpy as np

bc = BertClient()
result = []
value = 0.90
f = open('./data/train.txt', 'r')
for line in f:
    result.append(line.strip('\n'))

Input = bc.encode(result)
print(Input)

np.savetxt("./data/vector.txt",Input)