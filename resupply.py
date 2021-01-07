"dependencies:"

from itertools import permutations, product

from numpy import zeros

from math import factorial

"_________________________________________________________________________________________"


def expandPartitionList(inputPartitionList, newterm):
    newPartitionList = []
    for partition in inputPartitionList:
        for i in range(len(partition)):
            newpartition = []
            for j in range(len(partition)):
                if i == j: newpartition.append(partition[j] + [newterm])
                else: newpartition.append(partition[j])
            newPartitionList.append(newpartition)
        newPartitionList.append(partition + [[newterm]])
    return newPartitionList

def listOfPartitions(numberOfElements):
    partitionList = [[[1]]]
    for i in range(2,numberOfElements+1): partitionList = expandPartitionList(partitionList, i)
    return partitionList

def fullList(partitionList):
    fullList = []
    for partition in partitionList:
        permutationListNested = []
        for set in partition:
            permutationListNested.append(list(permutations(set)))
        fullList += list(product(*permutationListNested))
    return fullList
    
def solutionSpace(F): return fullList(listOfPartitions(F))

def constructRouteMatrix(soln, L, X):
    size = len(X) + 1
    N = zeros((size, size))
    for perm in soln:
        leftovers = L
        returnHome = 1
        for i in range(len(perm)-1):
            P = X[perm[i]-1]
            if returnHome == 1:
                N[0][perm[i]] += 1
            if leftovers > P:
                leftovers -= P
                restocks = 0
                returnHome = 0
            else:
                P -= leftovers
                if P%L == 0:
                    restocks = P//L
                    returnHome = 1
                    leftovers = L
                else:
                    restocks = P//L + 1
                    returnHome = 0
                    leftovers = L - P%L
            N[0][perm[i]] += restocks
            N[perm[i]][0] += restocks
            if returnHome == 1:
                N[perm[i]][0] += 1
            else:
                N[perm[i]][perm[i+1]] += 1
        P = X[perm[-1]-1]
        if returnHome == 1:
            N[0][perm[-1]] += 1
        if leftovers > P:
            restocks = 0
        else:
            P -= leftovers
            restocks = (P-1)//L + 1
        N[0][perm[-1]] += restocks
        N[perm[-1]][0] += restocks + 1
    return N

def cost(N, C):
    itterations = range(len(C))
    cost = 0
    for i in itterations:
        for j in itterations:
            cost += N[i][j] * C[i][j]
    return cost

def constructQualityVector(solutionSpace, L, X, C):
    q = []
    for soln in solutionSpace:
        q.append(cost(constructRouteMatrix(soln, L, X) , C))
    return q

def solutionQualityByIndex(index, F, L, X, C):
    quality = cost(constructRouteMatrix(solutionSpace(F)[index], L, X) , C)
    return quality

def getPermCode(permutation):
    depth = len(permutation)
    ordered = sorted(permutation)
    code = []
    for i in permutation:
        edge = ordered.index(i)
        code.append(edge)
        del(ordered[edge])
    return code

def getPermIndex(permutation):
    code = getPermCode(permutation)
    maxDepth = len(code)
    order = 0
    while len(code) > 0:
        order += factorial(maxDepth - len(code))*code.pop(-1)
    return order

def getIndexCombinedPerm(comPerm):
    if len(comPerm) == 1: return getPermIndex(comPerm[0])
    baseindex = getPermIndex(comPerm[0])
    for perm in comPerm[1:]:
        baseindex *= factorial(len(perm))
    return baseindex + getIndexCombinedPerm(comPerm[1:])

def getSize(partition):
    size = 0
    for group in partition:
        size += len(group)
    return size

def getPartitionCode(partition):
    size = getSize(partition)
    groups = len(partition)
    code = []
    for element in range(2, size + 1):
        for group in range(groups):
            if element in partition[group]:
                code.append(group)
    return code

def permutationsAtEndpoint(code):
    count = 1
    newcode = [i for i in code]
    newcode.insert(0,0)
    groupcounts = []
    for i in range(max(newcode)+1):
        groupcounts.append(newcode.count(i))
    for j in groupcounts:
        count *= factorial(j)
    return count

def downstreamPermCount(code, maxDepth):
    count = 0
    if len(code) == maxDepth: return permutationsAtEndpoint(code)
    branches = max(code) + 2
    for i in range(branches):
        count += downstreamPermCount(code + [i], maxDepth)
    return count

def getSolutionIndex(solution):
    partitionCode = getPartitionCode(solution)
    maxPartitionDepth = len(partitionCode)
    index = getIndexCombinedPerm(solution) 
    while len(partitionCode) > 0:
        for i in range(partitionCode[-1]):
            partitionCode[-1] = i
            index += downstreamPermCount(partitionCode, maxPartitionDepth)
        del(partitionCode[-1])
    return index

def generatePartitionCodeAndPermIndex(F, index):
    if index > maxIndex(F):
        print("choose a lower index")
        return
    maxDepth = F - 1
    count = downstreamPermCount([0], maxDepth)
    if index > count - 1:
        partitionCode = [1]
        index -= count
    else:
        partitionCode = [0]
    while len(partitionCode) < maxDepth:
        branches = max(partitionCode) + 2
        for i in range(branches):
            count = downstreamPermCount(partitionCode + [i], maxDepth)
            if count > index:
                partitionCode.append(i)
                break
            else:
                index -= count
    return partitionCode, index

def maxIndex(F):
    maxPartitionCode = []
    for i in range(1, F):
        maxPartitionCode.append(i)
    maxPartitionDepth = F-1
    index = 0 
    while len(maxPartitionCode) > 0:
        for i in range(maxPartitionCode[-1]):
            maxPartitionCode[-1] = i
            index += downstreamPermCount(maxPartitionCode, maxPartitionDepth)
        del(maxPartitionCode[-1])
    return index

def generatePartition(partitionCode):
    partition = [[1]]
    for i in range(len(partitionCode)):
        if partitionCode[i] > len(partition) - 1:
            partition.append([i+2])
        else: partition[partitionCode[i]].append(i+2)
    return partition

def generatePermCodeFromIndex(index, setSize):
    maxDepth = setSize
    permCode = []
    while len(permCode) < maxDepth:
        branches = maxDepth - len(permCode)
        PermsPerBranch = factorial(branches - 1) 
        permCode.append(index//PermsPerBranch)
        index = index%PermsPerBranch
    return permCode

def PermFromCode(permCode, set):
    permutation = []
    for i in permCode:
        permutation.append(set.pop(i))
    return permutation

def comIndexFromIndex(index, partition):
    groupSizes = []
    for i in partition:
        groupSizes.append(len(i))
    size = len(groupSizes)
    comIndex = []
    for i in range(size-1):
        product = 1
        for j in range(i+1, size):
            product *= factorial(groupSizes[j])
        comIndex.append(index//product)
        index = index%product
    comIndex.append(index)
    return comIndex

def comPermFromIndex(index, partition):
    comPerm = []
    comIndex = comIndexFromIndex(index, partition)
    for i in range(len(comIndex)):
        comPerm.append(PermFromCode(generatePermCodeFromIndex(comIndex[i], len(partition[i])),partition[i]))
    return comPerm

def generateSolutionFromIndex(index, F):
    partitionCode, comPermIndex = generatePartitionCodeAndPermIndex(F, index)
    partition = generatePartition(partitionCode)
    return comPermFromIndex(comPermIndex, partition)
