import sys
import pandas as pd
import numpy as np
import os

f_train, f_test, f_result = sys.argv[1:4] #argument

dt_train = pd.read_csv(f_train, sep='\t')

def getInfo(D): # Info(D) = -sum(1->m) (p(i)*log2(p(i))
  m, counts = np.unique(D.iloc[:, -1], return_counts = True)
  return -np.sum( np.fromiter( (counts[i]/len(D) * np.log2(counts[i] / len(D)) for i in range(len(m))), float ) )
 
def getGain(D, attr): #InfoA(D) = sum(1->v) |D(j)|/|D| * Info(D(j))
  v, counts = np.unique(D[attr], return_counts = True) # v = attribute의 class# 
  return getInfo(D) - np.sum( np.fromiter( (counts[j] / len(D) * getInfo(D.query(f"{attr} == '{v[j]}'")) for j in range(len(v))), float ) )

def getGainRatio(D, attr): #GainRatio = Gain(A) / SplitInfo(A)
                           #SpitInfoA(D) = -sum(1->v) |D(j)|/|D| * log2(|D(j)|/|D|)
  v, counts = np.unique(D[attr], return_counts = True)
  splitInfo = - np.sum( np.fromiter( (counts[j] / len(D) * np.log2(counts[j] / len(D)) for j in range(len(v)) ), float  ) )
  return getGain(D, attr) / splitInfo

def getCandidate(D): #get max-gain attribute name
  gains = np.array([getGainRatio(D, attr) for attr in D.columns[:-1]])
  return D.columns[gains.argmax()]

Tree_Node = {
    'Candidate' : '',
    'Attribute' : '',
    'Childs' : {},
}

def getDecisionTree(D, Node): #D는 samples / attr은 나눌 attribute

  New_Node = {
    'Candidate' : '',
    'Attribute' : '',
    'Childs' : {},
  }
  if D.size == 0: #더이상 남은 샘플이 없을 때
    return None

  elif D.keys().size == 1: #더 이상 나눌 수 있는 attribute가 없을 때
    classes, counts = np.unique(D, return_counts=True) 
    return classes[counts.argmax()]
  
  elif len(np.unique(D.values[:,-1:])) == 1: #남은 샘플의 클래스가 동일할 때
    return np.unique(D.values[:,-1:])[0]

  else:
    
    Node['Candidate'] = getCandidate(D)
    New_Node['Attribute'] = Node['Candidate']
    classes = np.unique(D[Node['Candidate']]) #attribute의 classes
    for cls in classes:
      Child_Node = New_Node.copy() #child노드 생성
      if len(Child_Node['Childs']) != 0:
        Child_Node['Childs'] = {}
      New_D = D.query(f"{Node['Candidate']} == '{cls}'").drop(Node['Candidate'], axis=1) #data split
      (Node['Childs'])[cls] = (getDecisionTree(New_D, Child_Node),len(New_D)) #node와 node의 길이 set
    
    #Node['Childs'] = childs #부모노드에 자식노드들 추가
    
  return Node

Tree_Node = getDecisionTree(dt_train, (Tree_Node))

#pprint(Tree_Node)

def getPredict(DTree, sample):
  if type(DTree) is tuple and type(DTree[0]) is not dict: #leaf node
    return DTree[0]
  else:
    if type(DTree) is tuple:
      DTree = DTree[0]
    if sample[DTree['Candidate']] in DTree['Childs'].keys(): #해당 attribute 값이 tree에 있다면
      return getPredict(DTree['Childs'][sample[DTree['Candidate']]], sample)
    else: #없다면
      max_key = max(DTree['Childs'], key = (lambda k: DTree['Childs'][k][1]))
      return getPredict(DTree['Childs'][max_key], sample)

    


dt_test = pd.read_csv(f_test, sep='\t')
add_row = []

for i, row in dt_test.iterrows():
  add_row.append(getPredict(Tree_Node, row))

dt_test[dt_train.keys()[-1]] = np.array(add_row)

dt_test.to_csv(f_result, mode='w', sep='\t', index=False)

