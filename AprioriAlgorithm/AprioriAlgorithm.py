import sys

min_sup, path_input, path_output = sys.argv[1:4]
min_sup = int(min_sup)

f_input = open(path_input, "r")
lines = f_input.read().splitlines()
f_input.close()

item_num = 0
trans = []

for line in lines:
    items = [int(i) for i in line.split('\t')]
    trans.append(items)
item_num = max(map(max, trans)) + 1

total_trans_num = len(trans)

#print (item_num)

def find_support(itemset, transactions): #transactions 중 itemset이 차지하는 비율
    sup_count = 0
    for transaction in transactions:
        if set(transaction).issuperset(set(itemset)): #transation이 itemset을 포함할 경우
            sup_count += 1
    return sup_count/total_trans_num # support = sup_count / 전체 거래 수

def find_confidence(x, y, transactions):
    return find_support(x|y,transactions)/find_support(x,transactions)


### 조합으로 찾기 실패 - 계산량 너무 많음
#### candidate의 합집합의 집합으로
#pruned_sets 대신 frequent_item_sets으로

'''
def get_candidates(itemsets, n, pruned_sets):#candidates를 생성하는 generator
    for i in range(len(itemsets)):  
        if n == 1:
            yield [itemsets[i]]
        else:
            for next in get_candidates(itemsets[i+1:], n-1, pruned_sets):
                items = [itemsets[i]] + next
                if not pruned_sets: #비어있으면
                    yield items
                pruned_check = False
                for pruned_set in pruned_sets:
                    if set(items).issuperset(set(pruned_set)): #items가 pruned_sets 중 하나의 superset이라면
                        pruned_check = True #items가 제외되었기 때문에 yield하지 않고 다음 후보로 넘어감
                if pruned_check:
                    continue
                else:
                    yield items
'''

def Apriori(candidates, pruned_sets):
    new_candidates = []
    new_pruned_sets = []
    for i in range(len(candidates)):
        itemset = set(candidates[i]) #support를 계산할 itemset
        for j in range(i+1, len(candidates)):
            #import pdb; pdb.set_trace()
            original_itemset = itemset.copy()
            itemset = itemset | set(candidates[j]) #중복 제거한 set
            
            if itemset in new_candidates or itemset in new_pruned_sets: #이미 검사된 itemset이라면 패스
                continue
            
            pruned_check = False #pruned 여부
            for pruned_set in pruned_sets:
                if itemset.issuperset(set(pruned_set)): #itemset가 pruned_sets 중 하나의 superset이라면
                    pruned_check = True
                    break
            if pruned_check:
                continue #pruned된 itemset이면 no check
            
            if find_support(itemset, trans) >= min_sup/100:
                new_candidates.append(itemset.copy())
            else:
                new_pruned_sets.append(itemset.copy())
            itemset = original_itemset #원래대로

    ''' 
    for candidate in new_candidates:
        print(f"{candidate}\t{find_support(candidate,trans)}\n")
    '''
        
    if(new_candidates == []): #모든 candidate가 pruned되었을 때
        #import pdb; pdb.set_trace()
        return candidates
    else:
        #import pdb; pdb.set_trace()
        return Apriori(new_candidates, new_pruned_sets) + candidates



def get_rule_sets(iterable):
    from itertools import chain, combinations
    s = list(iterable)
    powerset = list(chain.from_iterable(combinations(s, r) for r in range(1, len(s))))
    return [(set(powerset[i]), set(powerset[len(powerset) - 1 - i])) for i in range(len(powerset))]


items = list(range(item_num)) #기본 items

frequent_item_sets = [[item] for item in items if find_support([item],trans) >= min_sup/100] #1-itemsets
frequent_item_sets = [sorted(list(sets)) for sets in Apriori(frequent_item_sets, []) if list(sets) not in frequent_item_sets]
#print(frequent_item_sets)

f_output = open(path_output, "w")

i = 0
for frequent_item_set in frequent_item_sets:
    for rule in get_rule_sets(frequent_item_set):
        #f_output.write(f"{rule[0]}\t{rule[1]}\t{round(find_support(frequent_item_set,trans)*100,2)}\t{round(find_confidence(rule[0],rule[1],trans)*100,2)}\n")
        print(rule[0],rule[1],format(find_support(frequent_item_set,trans)*100,".2f"),format(find_confidence(rule[0],rule[1],trans)*100,".2f"),sep='\t',file=f_output)



