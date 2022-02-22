import csv
import tqdm
from collections import defaultdict

def calculateRoute(lst,path):
    """
    The function find the route between two cities using A*-search algorithm.

    Example:
    calculateRoute([Istanbul,Ankara], “distances.csv”)
    Output : Istanbul-Izmit-Bolu-Sakarya-Ankara,468 
    
    Parameters:
    lst(list) : Containing start and stop of city
    path(str) : File path in memory
    
    Return:
    string : The short routh of start between stop and total distance. 
    
    """
    if not path.endswith('csv'): raise TypeError('Check your file type. It  is not .csv format') #check data is csv format
    
    start_node,stop_node = lst #Start ans Stop node are defined
    graph,heuristic = create_graph(path) #undirected weighted graph define,create heuristic distances each cities
    
    open_set = list()
    open_set.append(start_node)
    closed_set = list()

    g = {} #store distance from starting node
    parents = {}# parents contains an adjacency map of all nodes
    g[start_node] = 0 #ditance of starting node from itself is zero
    parents[start_node] = (start_node,0) #start_node is start node so parent node is itself
    
    def get_neigbours(v):
        """
        The function  check if  parent node has neighbours.

        Parameters:
        v(string) : Parent node name

        Return:
        If node has neigbours return neigbours. If it has not return None 
        
        """
        if v in graph:
            return graph[v]
        else :
            return None
    
    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n == None or g[v]+heuristic[v] < g[n]+heuristic[n]:
                n = v
        if n == stop_node or graph[n]==None:
            pass
        else:
            for(m,weight) in get_neigbours(n):
                if m not in open_set and m not in closed_set:
                        open_set.append(m)
                        parents[m] = (n,weight)
                        g[m] = g[n] + weight
                        
                else:
                    if g[m] > g[n] + weight:
                        #update g(m)
                        g[m] = g[n] + weight
                        #change parent of m to n
                        parents[m] = (n,weight)
                        
                            
                        #if m in closed set,remove and add to open
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.append(m) 
        if n == None:
            print('Path does not exist!')
            return None
 
        # if the current node is the stop_node
        # then we begin reconstructin the path from it to the start_node
        if n == stop_node:
            path_way = []
            total_distance = 0
            while parents[n][0] != n:
                path_way.append(n)
                total_distance += parents[n][1]
                n = parents[n][0]

            path_way.append(start_node)

            path_way.reverse()
            total_distance = total_distance
            print('-'.join(path_way)+','+'{}'.format(total_distance))
            return "-".join(path_way)
        open_set.remove(n)
        closed_set.append(n)


def create_graph(path):
    """
    The function read CSV format and create graph from data that collected CSV file
    
    Parameters:
    path(str) : File path in memory

    Return:
    H_dict(dictionary) :The heuristic approximation of the value of the node
    graph(dictionary)  :The undirected weighted graph 
    """
    graph = defaultdict(list)#provides a default value for the key that does not exists.
    H_dict = {}
    file =open(path)
    csvread=csv.reader(file,delimiter=',')
    for row in csvread:
        H_dict[row[0]]=1
        H_dict[row[1]]=1
        a,b,c = row[0],row[1],row[2]
        graph[a].append((b,int(c)))
        graph[b].append((a,int(c)))
    
    return graph,H_dict




if __name__ == '__main__':  
    calculateRoute(['Istanbul','Ankara'],'distances.csv')