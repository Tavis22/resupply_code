from numpy import savetxt

from resupply import solutionSpace, constructQualityVector

def main():
    F = 6
    C = [[0,18,18,15,13,16,17],[18,0,3,2,3,10,8],[18,3,0,6,6,11,11],[15,2,6,0,15,15,8],[13,3,6,15,0,7,5],[16,10,11,15,7,0,14],[17,8,11,8,5,14,0]]
    X = [18,24,6,14,23,15]
    L = 20
    solutions = solutionSpace(F)
    q = constructQualityVector(solutions, L, X, C)
    savetxt("6nodeQualityVector.csv", q, delimiter=",")

main()