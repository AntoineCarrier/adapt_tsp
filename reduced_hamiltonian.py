
def adapt_tsp_hamiltonian(N, B, W, graph, fqubit):
    """
    Function to compute the tsp hamiltonian terms, 
    adapted to the adapt-Clifford algorithm.

    N = number of nodes in the graph
    B = energy hierarchy coefficient
    W = the coupling coefficients matrix
    graph = the graph on which to compute the hamiltonian cycle    
    """
    for i in range(N**2):
        if i==fqubit:
            continue
        W[i][N**2] += 1
        W[N**2][i] += 1


    for r in range(N):
        for i in range(1, N):
            for j in range(i):

                # if i==fqubit or j==fqubit:
                    # continue
                W[i * N + r][N**2] += -1/2
                W[N**2][i * N + r] += -1/2

                W[j * N + r][N**2] += -1/2
                W[N**2][j * N + r] += -1/2

                W[i * N + r][j * N + r] += 1/2
                W[j * N + r][i * N + r] += 1/2     
       

    for i in range(N):
        for r in range(1, N):
            for s in range(r):
                # if i==fqubit:
                    # continue
                W[i * N + r][N**2] += -1/2
                W[N**2][i * N + r] += -1/2
            
                W[i * N + s][N**2] += -1/2
                W[N**2][i * N + s] += -1/2

                W[i * N + r][i * N + s] += 1/2
                W[i * N + s][i * N + r] += 1/2

    for i in range(N):
        for j in range(N):
            for r in range(N):
                if i == j:
                    continue 
                s = (r + 1)%N
                if i*N+r == j * N + s:
                    continue

                if i==fqubit or r==fqubit or j==fqubit or s==fqubit:
                    continue

                if (i, j) in graph.edges:
                    DuA = -B * graph.edges[i, j]['weight'] * 1/4
                    Duv = B * graph.edges[i, j]['weight'] * 1/4
                    W[i * N + r][N**2] += DuA
                    W[N**2][i * N + r] += DuA
                
                    W[j * N + s][N**2] += DuA
                    W[N**2][j * N + s] += DuA
                
                
                    W[i * N + r][j * N + s] += Duv
                    W[j * N + s][i * N + r] += Duv
                
                elif (j, i) in graph.edges:
                    DuA = -B * graph.edges[j, i]['weight'] * 1/4
                    Duv = B * graph.edges[j, i]['weight'] * 1/4
                    W[i * N + r][N**2] += DuA
                    W[N**2][i * N + r] += DuA
                
                    W[j * N + s][N**2] += DuA
                    W[N**2][j * N + s] += DuA
                
                
                    W[i * N + r][j * N + s] += Duv
                    W[j * N + s][i * N + r] += Duv   

    for i in range(N):
        if i == fqubit:
            continue
        if (i, fqubit) in graph.edges:
            DuA = -B * graph.edges[i, fqubit]['weight'] * 1/4
            Duv = B * graph.edges[i, fqubit]['weight'] * 1/4
            W[i * N + 1][N**2] += DuA
            W[N**2][i * N + 1] += DuA

            W[fqubit * N + s][N**2] += DuA
            W[N**2][fqubit * N + s] += DuA


            W[i * N + 1][fqubit * N + s] += Duv
            W[fqubit * N + s][i * N + 1] += Duv

        elif (fqubit, i) in graph.edges:
            DuA = -B * graph.edges[fqubit, i]['weight'] * 1/4
            Duv = B * graph.edges[fqubit, i]['weight'] * 1/4
            W[i * N + 1][N**2] += DuA
            W[N**2][i * N + 1] += DuA

            W[fqubit * N + s][N**2] += DuA
            W[N**2][fqubit * N + s] += DuA


            W[i * N + 1][fqubit * N + s] += Duv
            W[fqubit * N + s][i * N + 1] += Duv   

    

    return W

