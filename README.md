# Traveling Salesman Problem with a Drone - TSP-D
**Authors**: Chile Ovidiu-Benone MOC2, Pintilie Andrei MOC1

_AEA2019_



**The Traveling Salesman Problem with a Drone (TSP-D)** is an extension of the Traveling Salesman Problem (TSP – also known as Traveling Salesman Person). In the TSP problem, a truck has to deliver a parcel to _n_ houses, and must do it so by only visiting each house once. Basically, it is the same as finding the shortest Hamiltonian cycle of an undirected graph.

The TSPd problem introduces the concept of a drone which helps with the delivery: the drone can get a single parcel from the truck and deliver it to one of the houses; after doing so, it must go back to the truck to get another parcel. This synchronization between the truck and the drone can only be done at one of the house.

The article at https://www.andrew.cmu.edu/user/vanhoeve/papers/tsp_drone_cpaior2019.pdf provides a solution to this problem using the Constraint Programming (CP) method, using Dynamic Programming (DP) and using Branch and Bound (BAB). Although the models and algorithms given could be better detailed, the authors (Ziye Tang, Willem-Jan van Hoeve1, and Paul Shaw) defined and proved the formalities behind the problem, providing also complexity analysis and some examples.

The problem to solve was TSPD. As this problem is known to be NP-hard, a deterministic approach for this problem would give either a far from the optimum solution, either a way too much time consuming solution. For this reason, we have implemented 2 heuristic solutions to solve our problem: a Genetic Algorithm (GA) and an Ant Colony Optimization (ACO or simply AC).
