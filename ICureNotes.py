import numpy as np
import pandas as pd
#Pip install fiass-cpu
import faiss
#Make Random vectors
data = np.random.random((1000,128)).astype('float32')
#Make random vectors with int from 1 to 100
data = np.random.randint(1,101,size=(1000,128)).astype('float32')

#df = pd.DataFrame(data)
#df.to_csv("Vectors.csv",index=False)

vector = np.random.randint(1,101,size=(1,128))
print(vector)
data = pd.read_csv("Vectors.csv",index_col=False).to_numpy()

index = faiss.IndexFlatL2(128)
index.add(data)
distances,indices = index.search(vector,k=1)
print(data[indices[0]])