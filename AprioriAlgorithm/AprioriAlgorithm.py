import sys

min_sup, path_input, path_output = sys.argv[1:4] #argument
min_sup = int(min_sup)/100 #Minimum Support

f_input = open(path_input, "r")
lines = f_input.read().splitlines()
f_input.close()
item_num = 0
trans = []

for line in lines:
    items = [int(i) for i in line.split('\t')]
    trans.append(items)

item_num = max(map(max, trans)) + 1 #item 종류(transaction에 있는 item 종류 중 최대값)

total_trans_num = len(trans) #전체 transaction의 길이

#support = 전체 transactions에서 itemset이 포함된 transaction의 비율
def find_support(itemset): 
    sup_count = 0
    for transaction in trans:
        if set(transaction).issuperset(set(itemset)): #transation이 itemset을 포함할 경우
            sup_count += 1
    return sup_count/total_trans_num # support = sup_count / 전체 거래 수


#confidence = X->Y의 Association Rule에서 X를 가진 transaction이 Y도 가질 조건적확률
def find_confidence(rule): 
    return find_support(rule[0]|rule[1])/find_support(rule[0]) #계산은 Sup(X,Y)/Sup(X)


#실제 Apriori 구현
def Apriori(candidates, pruned_sets):

    #next sets initialize
    next_candidates = []
    next_pruned_sets = []

    for i in range(len(candidates)):
        candidate = set(candidates[i])          #support를 계산할 candidate
        
        for j in range(i+1, len(candidates)):
            
            original_candidate = candidate.copy()         #계산 후 복구할 original candidate
            candidate = candidate | set(candidates[j])  #합집합 set

            #1. 이미 검사된 candidate이라면 패스
            if candidate in next_candidates or candidate in next_pruned_sets: 
                continue
            
            #2. candidate가 pruned_sets 중 하나의 superset이라면 패스
            pruned_check = False #pruned 여부
            for pruned_set in pruned_sets:
                if candidate.issuperset(set(pruned_set)): 
                    pruned_check = True
                    break
            if pruned_check:
                continue

            #3. 
            if find_support(candidate) >= min_sup: #Minimum Support를 만족하면 next_candidates로 copy
                next_candidates.append(candidate.copy())
            else:                                  #만족하지 못하면 next_pruned_sets로 copy
                next_pruned_sets.append(candidate.copy())
                
            candidate = original_candidate #원래값으로

    if(next_candidates == []):#모든 candidate가 pruned되어 next_candidate가 생성되지 않을 때
        return candidates
    else:                     #new_candidates가 있다면 recursive하게 candidates를 추가함
        return Apriori(next_candidates, next_pruned_sets) + candidates 


#itemsets에서 association rules set을 가져오는 함수
def get_rule_sets(itemsets):                
    from itertools import chain, combinations
    s = list(itemsets)
    
    #get Powerset of itemsets
    powerset = list(chain.from_iterable(combinations(s, r) for r in range(1, len(s))))
    
    #대응하는 subsets끼리 묶어서 return
    return [(set(powerset[i]), set(powerset[len(powerset) - 1 - i])) for i in range(len(powerset))] 


items = list(range(item_num)) #기본 items

frequent_itemsets = [[item] for item in items if find_support([item]) >= min_sup] #1-itemsets
frequent_itemsets = [sorted(list(sets)) for sets in Apriori(frequent_itemsets, []) if list(sets) not in frequent_itemsets]

f_output = open(path_output, "w")

i = 0
for frequent_itemset in frequent_itemsets:
    for rule in get_rule_sets(frequent_itemset):
        print(rule[0],rule[1],format(find_support(frequent_itemset)*100,".2f"),format(find_confidence(rule)*100,".2f"),sep='\t',file=f_output)

f_output.close()

