16 Feb 2022

**Wtech AI Course - Homework -1**

**Deadline: 23 February 2022![](Aspose.Words.90728727-5d32-4e1b-9c3e-677d2f105986.001.png)**

**Submission:** Submit your code/ jupyter notebook to [mursel.tasgin@gmail.com ](mailto:mursel.tasgin@gmail.com)**Other:** Open a github account on github.com and create a repository. Put your work also on your github repo, add me as a contributor to your repo (my github profile is murselTasginBoun)

**Howework Details:![](Aspose.Words.90728727-5d32-4e1b-9c3e-677d2f105986.002.png)**

Write a python program to find the route between two cities using A\*-search algorithm. Details:

- **Program will get 2 inputs:**
  - 1st input:  list: [from\_city, to\_city] ,  i,e:  [Istanbul,Ankara]
  - 2nd input: string: “distances.csv” → File name containing containing connections and distances between cities
  - Program call will look like: calculateRoute([Istanbul,Ankara], “distances.csv”)
- **Program will print the route (separated by a single dash (“-”) and total distance: Output:**Istanbul-Izmit-Bolu-Sakarya-Ankara,458
- The .csv file  (i.e., *distances.csv*) containing distances between cities has the following format. Each line in the file will show source-city name, destination-city name and distance. Distances are the same in both direction (i.e., Istanbul-Izmit distance is 103 km and Izmit-Istanbul distance is 103 km).

The file will look like this:

**distances.csv** Istanbul,Izmit,103 Izmit,Adapazari,53 Adapazari,Bolu,125 Bolu,Ankara,187 Ankara,Kirikkale,73 Kirsehir,Kayseri,135 Isparta,Antalya,126

- You should represent the distances in an undirected weighted graph (or tree)
- Program should be able find distance and route between any reachable cities.
