from ortools.sat.python import cp_model

houses = ["House #1", "House #2", "House #3", "House #4", "House #5"]

colours = ["red", "green", "ivory", "yellow", "blue" ]
nationalities = ["English", 
                 "Spanish",
                 "Ukrainian", 
                 "Norwegian", 
                 "Japanese"]
pets = ["dog", "snails", "fox", "horse", "zebra"]
drinks = ["coffee", "tea", "milk", "juice", "water"]
cigarettes = ["Old Gold", 
              "Chesterfields", 
              "Kools", 
              "Lucky Strike", 
              "Parliaments"]


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, colour, nationality, pet, drink, cigarette):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.colour_ = colour
        self.nationality_ = nationality
        self.pet_ = pet
        self.drink_ = drink
        self.cigarette_ = cigarette
        self.solutions_ = 0

    def OnSolutionCallback(self):
        self.solutions_ = self.solutions_ + 1
        print("solution", self.solutions_ )
        
        for house in houses:
            print(" - "+house+":")
            for colour in colours:
                if (self.Value(self.colour_[house][colour])):
                    print("    - ", colour)
            for nationality in nationalities:
                if (self.Value(self.nationality_[house][nationality])):
                    print("    - ", nationality)
            for pet in pets:
                if (self.Value(self.pet_[house][pet])):
                    print("    - ", pet)
            for drink in drinks:
                if (self.Value(self.drink_[house][drink])):
                    print("    - ", drink)
            for cigarette in cigarettes:
                if (self.Value(self.cigarette_[house][cigarette])):
                    print("    - ", cigarette)
        
        print()
        
def main():
    model = cp_model.CpModel()

    house_colour = {}
    for house in houses:        
        variables = {}
        for colour in colours:    
            variables[colour] = model.NewBoolVar(house+colour)
        house_colour[house] = variables
    
    house_nationality = {}
    for house in houses:        
        variables = {}
        for nationality in nationalities:    
            variables[nationality] = model.NewBoolVar(house+nationality)
        house_nationality[house] = variables

    house_pet = {}
    for house in houses:        
        variables = {}
        for pet in pets:    
            variables[pet] = model.NewBoolVar(house+pet)
        house_pet[house] = variables

    house_drink = {}
    for house in houses:        
        variables = {}
        for drink in drinks:    
            variables[drink] = model.NewBoolVar(house+drink)
        house_drink[house] = variables

    house_cigarette = {}
    for house in houses:        
        variables = {}
        for cigarette in cigarettes:    
            variables[cigarette] = model.NewBoolVar(house+cigarette)
        house_cigarette[house] = variables


    # every house has a different property
    for i in range(5):
        for j in range(i+1,5):
            for k in range(5):
                model.AddBoolOr([
                        house_colour[houses[i]][colours[k]].Not(), 
                        house_colour[houses[j]][colours[k]].Not()])
                model.AddBoolOr([house_nationality[houses[i]][nationalities[k]].Not(), house_nationality[houses[j]][nationalities[k]].Not()])
                model.AddBoolOr([house_pet[houses[i]][pets[k]].Not(), house_pet[houses[j]][pets[k]].Not()])
                model.AddBoolOr([house_drink[houses[i]][drinks[k]].Not(), house_drink[houses[j]][drinks[k]].Not()])
                model.AddBoolOr([house_cigarette[houses[i]][cigarettes[k]].Not(), house_cigarette[houses[j]][cigarettes[k]].Not()])
            
    for house in houses:
        # at least one property per house
        variables = []
        for colour in colours:
            variables.append(house_colour[house][colour])
        model.AddBoolOr(variables)

        variables = []
        for nationality in nationalities:
            variables.append(house_nationality[house][nationality])
        model.AddBoolOr(variables)

        variables = []
        for pet in pets:
            variables.append(house_pet[house][pet])
        model.AddBoolOr(variables)

        variables = []
        for drink in drinks:
            variables.append(house_drink[house][drink])
        model.AddBoolOr(variables)

        variables = []
        for cigarette in cigarettes:
            variables.append(house_cigarette[house][cigarette])
        model.AddBoolOr(variables)

        # max one property per house
        for i in range(5):
            for j in range(i+1,5):
                model.AddBoolOr([
                        house_colour[house][colours[i]].Not(), 
                        house_colour[house][colours[j]].Not()])
                model.AddBoolOr([
                        house_nationality[house][nationalities[i]].Not(), 
                        house_nationality[house][nationalities[j]].Not()])
                model.AddBoolOr([
                        house_pet[house][pets[i]].Not(), 
                        house_pet[house][pets[j]].Not()])
                model.AddBoolOr([
                        house_drink[house][drinks[i]].Not(), 
                        house_drink[house][drinks[j]].Not()])
                model.AddBoolOr([
                        house_cigarette[house][cigarettes[i]].Not(), 
                        house_cigarette[house][cigarettes[j]].Not()])
                
        # conditions
        model.AddBoolAnd([house_colour[house]["red"]]).OnlyEnforceIf(house_nationality[house]["English"])
        model.AddBoolAnd([house_pet[house]["dog"]]).OnlyEnforceIf(house_nationality[house]["Spanish"])
        model.AddBoolAnd([house_drink[house]["coffee"]]).OnlyEnforceIf(house_colour[house]["green"])
        model.AddBoolAnd([house_drink[house]["tea"]]).OnlyEnforceIf(house_nationality[house]["Ukrainian"])
        model.AddBoolAnd([house_pet[house]["snails"]]).OnlyEnforceIf(house_cigarette[house]["Old Gold"])
        model.AddBoolAnd([house_colour[house]["yellow"]]).OnlyEnforceIf(house_cigarette[house]["Kools"])
        model.AddBoolAnd([house_drink[house]["juice"]]).OnlyEnforceIf(house_cigarette[house]["Lucky Strike"])
        model.AddBoolAnd([house_cigarette[house]["Parliaments"]]).OnlyEnforceIf(house_nationality[house]["Japanese"])
    
    for i in range(1,4):
        model.AddBoolAnd([house_colour[houses[i+1]]["green"]]).OnlyEnforceIf(house_colour[houses[i]]["ivory"])
       
        model.AddBoolOr([
                house_pet[houses[i+1]]["fox"], 
                house_pet[houses[i-1]]["fox"]]).OnlyEnforceIf(house_cigarette[houses[i]]["Chesterfields"])
        model.AddBoolOr([
                house_pet[houses[i+1]]["horse"], 
                house_pet[houses[i-1]]["horse"]]).OnlyEnforceIf(house_cigarette[houses[i]]["Kools"])
        model.AddBoolOr([
                house_nationality[houses[i+1]]["Norwegian"], 
                house_nationality[houses[i-1]]["Norwegian"]]).OnlyEnforceIf(house_colour[houses[i]]["blue"])

    # Handle boundary cases
    model.AddBoolAnd([house_colour[houses[1]]["green"]]).OnlyEnforceIf(house_colour[houses[0]]["ivory"])
    model.AddBoolAnd([house_colour[houses[4]]["ivory"].Not()])

    model.AddBoolOr([house_pet["House #2"]["fox"]]).OnlyEnforceIf(house_cigarette["House #1"]["Chesterfields"])
    model.AddBoolOr([house_pet["House #4"]["fox"]]).OnlyEnforceIf(house_cigarette["House #5"]["Chesterfields"])

    model.AddBoolOr([house_pet[houses[1]]["horse"]]).OnlyEnforceIf(house_cigarette[houses[0]]["Kools"])
    model.AddBoolOr([house_pet[houses[3]]["horse"]]).OnlyEnforceIf(house_cigarette[houses[4]]["Kools"])

    model.AddBoolOr([house_nationality[houses[1]]["Norwegian"]]).OnlyEnforceIf(house_colour[houses[0]]["blue"])
    model.AddBoolOr([house_nationality[houses[3]]["Norwegian"]]).OnlyEnforceIf(house_colour[houses[4]]["blue"])

    # final conditions
    model.AddBoolAnd([house_drink["House #3"]["milk"]])
    model.AddBoolAnd([house_nationality["House #1"]["Norwegian"]])
       
    solver = cp_model.CpSolver()    
    solver.SearchForAllSolutions(model, SolutionPrinter(house_colour, house_nationality, house_pet, house_drink, house_cigarette))

    for house in houses:
        if solver.Value(house_drink[house]["water"]):
            for nationality in nationalities:
                if solver.Value(house_nationality[house][nationality]):
                    print("The "+nationality+" drinks water.")
        if solver.Value(house_pet[house]["zebra"]):
            for nationality in nationalities:
                if solver.Value(house_nationality[house][nationality]):
                    print("The "+nationality+" owns the zebra.")
                        

main()


