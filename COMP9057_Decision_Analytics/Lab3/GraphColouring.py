from ortools.sat.python import cp_model
from xml.dom import minidom
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from svg.path import parse_path
import copy

constaint_propagation = True
check_neighbours_of_neighbours = True
figure_delay1 = 1 # 1e-1
figure_delay2 = 1 # 1e-1

def load_map():    
    boundingbox = np.array([1e100,1e100,0,0])
    doc = minidom.parse("Counties_of_the_island_of_Ireland.svg")        
    transforms = {}
    for group in doc.getElementsByTagName("g"):
        transform = np.array(group.getAttribute("transform")[7:-1].split(","),dtype=np.float).reshape(3,2).T
        for path in group.getElementsByTagName("path"):
            county = path.getAttribute("id")
            transforms[county] = transform    
    county_maps = {}        
    for path in doc.getElementsByTagName("path"):
        county = path.getAttribute("id")
        county_maps[county] = []
        path_str = path.getAttribute("d")        
        path_str_split = path_str.split("z")        
        prev = complex(0,0)
        for polygon_path_str in path_str_split:
            polygon_path_str = polygon_path_str.strip()
            if len(polygon_path_str)>0:
                if polygon_path_str[0]=="m":
                    xy = polygon_path_str[2:polygon_path_str[2:].find(" ")+2].split(",")
                    point = prev + complex(float(xy[0]),float(xy[1]))
                    polygon_path_str = "m %s,%s"%(point.real,point.imag) + polygon_path_str[polygon_path_str[2:].find(" ")+2:]
                svg_path = parse_path(polygon_path_str)
                steps = max(int(svg_path.length(error=1e-5)/5), 5)
                
                points = []
                for i in range(steps+1):
                    point = svg_path.point(i/steps, error=1e-5)                            
                    if county in transforms:
                        transform = transforms[county]                            
                        xx = transform[0,0]*point.real + transform[0,1]*point.imag + transform[0,2]
                        yy = transform[1,0]*point.real + transform[1,1]*point.imag + transform[1,2]
                    else:
                        xx = point.real
                        yy = point.imag                    
                    points.append((xx,yy))
                    boundingbox[0] = min(boundingbox[0],xx)
                    boundingbox[1] = min(boundingbox[1],yy)
                    boundingbox[2] = max(boundingbox[2],xx)
                    boundingbox[3] = max(boundingbox[3],yy)
                prev = point
                county_maps[county].append(Polygon(points))
    doc.unlink() 
    return county_maps,boundingbox

def neighbourhood_graph():
    county_borders = pd.read_excel("County_borders.xlsx", index_col=0)
    counties = list(county_borders.columns)
    neighbours = {county:[] for county in counties}
    for county in counties:
        for other_county in county_borders[county][~pd.isna(county_borders[county])].keys():
            neighbours[county].append(other_county)
    return counties, neighbours

def colour_map(counties, colour_assignments, map_polygons):
    colour_mapping = {0:[0.7,0.7,0.7], 1:[1,0,0], 2:[0,1,0], 3:[0,0,1], 4:[1,1,0]}
    for county in counties:
        county_colour = colour_mapping[colour_assignments[county]]
        for poly in map_polygons[county]:
            poly.set_color(county_colour)
    plt.draw()
    plt.pause(figure_delay1)

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, counties, colour_assignments, map_polygons):
        cp_model.CpSolverSolutionCallback.__init__(self)
        #self.solutions_ = 0
        self.counties_ = counties
        self.colour_assignments_ = colour_assignments
        self.map_polygons_ = map_polygons
        return

    def OnSolutionCallback(self):
        #self.solutions_ = self.solutions_ + 1
        colour_map(self.counties_, {county:self.Value(self.colour_assignments_[county]) for county in self.counties_}, self.map_polygons_)
        # self.StopSearch()
        return

def find_colours_cpsat(counties, neighbours, map_polygons):
    model = cp_model.CpModel()
    colour_assignments = {}
    for county in counties:
        colour_assignments[county] = model.NewIntVar(1, 4, "colour_%s"%county)
    for county in counties:
        for neighbour in neighbours[county]:
            model.Add(colour_assignments[county] != colour_assignments[neighbour])
    solver = cp_model.CpSolver()
    solver.SearchForAllSolutions(model, SolutionPrinter(counties, colour_assignments, map_polygons))    
    return

def reduce_domain(domain, neighbour_domains):
    reduced = False
    for neighbour_domain in neighbour_domains:
        if len(neighbour_domain)==1:
            if neighbour_domain[0] in domain:
                domain.remove(neighbour_domain[0])
                reduced = True
    return reduced

def colour_counties_considered(current_county, counties, map_polygons):
    for poly in map_polygons[current_county]:
        poly.set_color([0.2,0.2,0.2])
    for county in counties:
        for poly in map_polygons[county]:
            poly.set_color([0.5,0.5,0.5])
    plt.draw()
    plt.pause(figure_delay2)

def propagate_constraints(current_county, domains, colour_assignments, neighbours, map_polygons):    
    if not constaint_propagation:
        return True
    # propagate neighbours
    queue = [neighbour for neighbour in neighbours[current_county] if colour_assignments[neighbour]==0]
    counties_considered = set(queue)
    while len(queue)>0:
        county = queue.pop()
        if reduce_domain(domains[county], [domains[neighbour] for neighbour in neighbours[county]]):
            if len(domains[county])==0:
                return False
            # propagate neighbours of neighbours
            if check_neighbours_of_neighbours:
                for neighbour in neighbours[county]:
                    if (not neighbour in queue) and colour_assignments[neighbour]==0:
                        counties_considered.add(neighbour)
                        queue.append(neighbour)
    colour_counties_considered(current_county, counties_considered, map_polygons)
    return True

def check_latest_assignment(current_county, colour_assignments, neighbours):
    for neighbour in neighbours[current_county]:
        if (colour_assignments[neighbour]>0) and (colour_assignments[neighbour] == colour_assignments[current_county]):
            return False        
    return True

def search(current_county, last_colour, colour_assignments, domains, counties, neighbours, map_polygons, no_of_nodes):    
    no_of_nodes += 1
    plt.title("%s"%no_of_nodes)
    colour_map(counties, colour_assignments, map_polygons)    
    # check if all counties are assigned a colour
    if current_county>=len(counties):
        return True, no_of_nodes
    current_domain = domains[counties[current_county]]
    for j in range(len(current_domain)):
        if last_colour in current_domain:
            # cycle through colours to distribute them evenly
            next_colour = current_domain[(current_domain.index(last_colour) + j + 1) % len(current_domain)]
        else:
            next_colour = current_domain[j]
        colour_assignments[counties[current_county]] = next_colour
        if check_latest_assignment(counties[current_county], colour_assignments, neighbours):
            reduced_domains = copy.deepcopy(domains)                    
            reduced_domains[counties[current_county]] = [next_colour]
            if not propagate_constraints(counties[current_county], reduced_domains, colour_assignments, neighbours, map_polygons):
                # backtrack because domain collapsed
                colour_assignments[counties[current_county]] = 0
            else:
                success,no_of_nodes = search(current_county+1, next_colour, colour_assignments, reduced_domains, counties, neighbours, map_polygons, no_of_nodes)
                if success:
                    return True,no_of_nodes
        else:
            # backtrack because assignment not feasible
            colour_assignments[counties[current_county]] = 0
    return False,no_of_nodes

def find_colours_search(counties, neighbours, map_polygons):    
    # sort counties to consider least number of neighbours first (this is the worst case, for educational purposes)
    counties.sort(key=lambda county:len(neighbours[county]))    
    colour_assignments = {county:0 for county in counties}    
    domains = {county:[1,2,3,4] for county in counties}
    success, no_of_nodes = search(0, 0, colour_assignments, domains, counties, neighbours, map_polygons, 0)
    return success

def draw_map(counties):
    county_maps,boundingbox = load_map()
    map_polygons = {county:[] for county in counties}
    plt.ion()
    plt.figure("Counties in Ireland")
    ax = plt.gca()
    ax.set_xlim(boundingbox[0],boundingbox[2])
    ax.set_ylim(boundingbox[3],boundingbox[1])
    for county in county_maps:
        if county in counties:
            for poly in county_maps[county]:
                map_polygons[county].append(poly)
                ax.add_patch(poly)                
    return map_polygons

def main():
    counties,neighbours = neighbourhood_graph()        
    map_polygons = draw_map(counties)

    find_colours_search(counties, neighbours, map_polygons)
    # find_colours_cpsat(counties, neighbours, map_polygons)
    
    plt.ioff()
    plt.show()

    return

main()
