#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This model is implementation of factory planning 2 problem [example #4]
   listed in fifth edition of Modeling Building in Mathematical Programming
   by H. P. Williams on pages 256 and 302 – 303
   Optimization model written below uses Google OR-Tools
"""

import itertools

from ortools.linear_solver import pywraplp

# =============================================================================
# Data
# =============================================================================

products = ["Prod1", "Prod2", "Prod3", "Prod4", "Prod5", "Prod6", "Prod7"]
machines = ["grinder", "vertDrill", "horiDrill", "borer", "planer"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

profit = {
    "Prod1": 10,
    "Prod2": 6,
    "Prod3": 8,
    "Prod4": 4,
    "Prod5": 11,
    "Prod6": 9,
    "Prod7": 3,
}

time_required = {
    "grinder": {"Prod1": 0.5, "Prod2": 0.7, "Prod5": 0.3, "Prod6": 0.2, "Prod7": 0.5},
    "vertDrill": {"Prod1": 0.1, "Prod2": 0.2, "Prod4": 0.3, "Prod6": 0.6},
    "horiDrill": {"Prod1": 0.2, "Prod3": 0.8, "Prod7": 0.6},
    "borer": {"Prod1": 0.05, "Prod2": 0.03, "Prod4": 0.07, "Prod5": 0.1, "Prod7": 0.08},
    "planer": {"Prod3": 0.01, "Prod5": 0.05, "Prod7": 0.05},
}


# number of machines down
machines_down = {"grinder": 2, "vertDrill": 2, "horiDrill": 3, "borer": 1, "planer": 1}

# number of each machine available
machines_installed = {
    "grinder": 4,
    "vertDrill": 2,
    "horiDrill": 3,
    "borer": 1,
    "planer": 1,
}

# market limitation of sells
max_sales = {
    ("Jan", "Prod1"): 500,
    ("Jan", "Prod2"): 1000,
    ("Jan", "Prod3"): 300,
    ("Jan", "Prod4"): 300,
    ("Jan", "Prod5"): 800,
    ("Jan", "Prod6"): 200,
    ("Jan", "Prod7"): 100,
    ("Feb", "Prod1"): 600,
    ("Feb", "Prod2"): 500,
    ("Feb", "Prod3"): 200,
    ("Feb", "Prod4"): 0,
    ("Feb", "Prod5"): 400,
    ("Feb", "Prod6"): 300,
    ("Feb", "Prod7"): 150,
    ("Mar", "Prod1"): 300,
    ("Mar", "Prod2"): 600,
    ("Mar", "Prod3"): 0,
    ("Mar", "Prod4"): 0,
    ("Mar", "Prod5"): 500,
    ("Mar", "Prod6"): 400,
    ("Mar", "Prod7"): 100,
    ("Apr", "Prod1"): 200,
    ("Apr", "Prod2"): 300,
    ("Apr", "Prod3"): 400,
    ("Apr", "Prod4"): 500,
    ("Apr", "Prod5"): 200,
    ("Apr", "Prod6"): 0,
    ("Apr", "Prod7"): 100,
    ("May", "Prod1"): 0,
    ("May", "Prod2"): 100,
    ("May", "Prod3"): 500,
    ("May", "Prod4"): 100,
    ("May", "Prod5"): 1000,
    ("May", "Prod6"): 300,
    ("May", "Prod7"): 0,
    ("Jun", "Prod1"): 500,
    ("Jun", "Prod2"): 500,
    ("Jun", "Prod3"): 100,
    ("Jun", "Prod4"): 300,
    ("Jun", "Prod5"): 1100,
    ("Jun", "Prod6"): 500,
    ("Jun", "Prod7"): 60,
}

inventory_holding_cost = 0.5
maximum_inventory = 100
inventory_target = 50
hours_per_month = 2 * 8 * 24

# model instantiation
model = pywraplp.Solver(
    "factory_planning_2", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
)

# =============================================================================
# # Decision variables:
# =============================================================================

month_product = list(itertools.product(months, products))

# 1.Number of units of every product to manufacture in each month
dv_manufacture = {i: model.NumVar(0, 1000000, "manuf_" + str(i)) for i in month_product}

# 2. Number of units of every product to store in each month
dv_inventory = {
    i: model.NumVar(0, maximum_inventory, "invt_" + str(i)) for i in month_product
}

# 3. Number of units of every product to sell in each month
dv_sold = {i: model.NumVar(0, max_sales[i], "sold_" + str(i)) for i in month_product}

# 4. Number of machines of scheduled for maintenance in each month
dv_repair = {
    (mth, mach): model.IntVar(
        0, machines_down[mach], "down_" + str(mth) + "_" + str(mach)
    )
    for mth in months
    for mach in machines
}


# =============================================================================
# # Constraints:
# =============================================================================

# 1. Initial Balance: For each product the number of units produced
#    should be equal to the number of units sold plus inventory
for prd in products:
    model.Add(
        dv_manufacture["Jan", prd] == dv_sold["Jan", prd] + dv_inventory["Jan", prd]
    )

# 2. Balance: For each product the number of units produced in each month
#    and the ones previously stored should be equal to the number of units
#    sold and inventory stored in that month
for mth in months[1:]:
    for prd in products:
        mth_index = months.index(mth)
        prev_mth = months[mth_index - 1]
        model.Add(
            dv_inventory[prev_mth, prd] + dv_manufacture[mth, prd]
            == dv_sold[mth, prd] + dv_inventory[mth, prd]
        )


# 3. Inventory Target: The number of units of product kept in inventory
#    at the end of the planning horizon should be equal to inventory target
for prd in products:
    model.Add(dv_inventory["Jun", prd] == inventory_target)

# 4. Machine Capacity: Total time used to manufacture any product at machine
#    cannot exceed its monthly capacity (in hours).
for mth in months:
    for mach in machines:
        model.Add(
            model.Sum(
                time_required[mach][prd] * dv_manufacture[mth, prd]
                for prd in time_required[mach]
            )
            <= hours_per_month * (machines_installed[mach] - dv_repair[mth, mach])
        )

# 5. The number of machines scheduled for maintenance should meet the
#    downtime requirement.
for mach in machines:
    model.Add(model.Sum(dv_repair[mth, mach] for mth in months) == machines_down[mach])

# =============================================================================
# # Objective function
# =============================================================================

obj_func = model.Sum(
    profit[prd] * dv_sold[mth, prd] for mth in months for prd in products
) - model.Sum(
    inventory_holding_cost * dv_inventory[mth, prd]
    for mth in months
    for prd in products
)


# solving the model
model.Maximize(obj_func)

status = model.Solve()

# test if solution is optimal
if status == pywraplp.Solver.OPTIMAL:
    print("Solution:")
    print("Objective value =", model.Objective().Value())
else:
    print("The problem does not have an optimal solution.")

print("\n")
print("Problem size:")
print("Number of decision variables =", model.NumVariables())
print("Number of constraints =", model.NumConstraints())

print("\n")
print("Advanced usage:")
print("Problem solved in %s seconds" % str(model.wall_time() / 1000))
print("Problem solved in %s iterations" % str(model.iterations()))
print("Problem solved in %d branch-and-bound nodes" % model.nodes())

solver_walltime_secs = model.wall_time() / 1000  # in seconds
