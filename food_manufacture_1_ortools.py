#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
   This model is implementation of food manufacture 1 problem [example #1]
   listed in fifth edition of Modeling Building in Mathematical Programming
   by H. P. Williams on pages 253 – 254 and 296 – 298.
   Optimization model written below uses Google OR-Tools
"""

import itertools

from ortools.linear_solver import pywraplp

# =============================================================================
# Data
# =============================================================================

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

oils = ["VEG1", "VEG2", "OIL1", "OIL2", "OIL3"]

cost = {
    ("Jan", "VEG1"): 110,
    ("Jan", "VEG2"): 120,
    ("Jan", "OIL1"): 130,
    ("Jan", "OIL2"): 110,
    ("Jan", "OIL3"): 115,
    ("Feb", "VEG1"): 130,
    ("Feb", "VEG2"): 130,
    ("Feb", "OIL1"): 110,
    ("Feb", "OIL2"): 90,
    ("Feb", "OIL3"): 115,
    ("Mar", "VEG1"): 110,
    ("Mar", "VEG2"): 140,
    ("Mar", "OIL1"): 130,
    ("Mar", "OIL2"): 100,
    ("Mar", "OIL3"): 95,
    ("Apr", "VEG1"): 120,
    ("Apr", "VEG2"): 110,
    ("Apr", "OIL1"): 120,
    ("Apr", "OIL2"): 120,
    ("Apr", "OIL3"): 125,
    ("May", "VEG1"): 100,
    ("May", "VEG2"): 120,
    ("May", "OIL1"): 150,
    ("May", "OIL2"): 110,
    ("May", "OIL3"): 105,
    ("Jun", "VEG1"): 90,
    ("Jun", "VEG2"): 100,
    ("Jun", "OIL1"): 140,
    ("Jun", "OIL2"): 80,
    ("Jun", "OIL3"): 135,
}


hardness = {"VEG1": 8.8, "VEG2": 6.1, "OIL1": 2.0, "OIL2": 4.2, "OIL3": 5.0}

price = 150
init_store_inventory = 500
target_store_inventory = 500
veg_upper_cap = 200
oil_upper_cap = 250

min_hardness = 3
max_hardness = 6
inventory_holding_cost = 5

# model instantiation
model = pywraplp.Solver("food_manufacture_1", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

# =============================================================================
# # Decision variables:
# =============================================================================

# 1. Tons of food to produce every month
dv_prod = {i: model.NumVar(0, 1000000, "prod_month_" + i) for i in months}

month_oil = list(itertools.product(months, oils))

# 2. Tons of oil to buy at month t
dv_oil_buy = {i: model.NumVar(0, 1000000, "buy_oil_" + str(i)) for i in month_oil}

# 3. Tons of oil o consumed every month
dv_oil_consume = {
    i: model.NumVar(0, 1000000, "consume_oil_" + str(i)) for i in month_oil
}

# 4. Tons of oil put in inventory every month
dv_oil_inventory = {
    i: model.NumVar(0, 1000000, "invt_oil_" + str(i)) for i in month_oil
}

# =============================================================================
# # Constraints:
# =============================================================================

# 1. Balance constraint for January
for ol in oils:
    model.Add(
        init_store_inventory + dv_oil_buy["Jan", ol]
        == dv_oil_consume["Jan", ol] + dv_oil_inventory["Jan", ol]
    )

# 2. Balance constraint for subsequent months
for mth in months[1:]:
    mth_index = months.index(mth)
    prev_mth = months[mth_index - 1]
    for ol in oils:
        model.Add(
            dv_oil_inventory[prev_mth, ol] + dv_oil_buy[mth, ol]
            == dv_oil_consume[mth, ol] + dv_oil_inventory[mth, ol]
        )

# 3. End of month inventory target
for ol in oils:
    model.Add(dv_oil_inventory["Jun", ol] == target_store_inventory)


# 4. Total Tons of each oil consumed in every month cannot exceed
#    the refinement capacity

# veg oils
veg_oils = ["VEG1", "VEG2"]
non_veg_oils = ["OIL1", "OIL2", "OIL3"]
for mth in months:
    model.Add(model.Sum(dv_oil_consume[mth, ol] for ol in veg_oils) <= veg_upper_cap)
    model.Add(
        model.Sum(dv_oil_consume[mth, ol] for ol in non_veg_oils) <= oil_upper_cap
    )

# 5.The hardness value of the food produced every month should be within tolerances.
for mth in months:
    model.Add(
        model.Sum(hardness[ol] * dv_oil_consume[mth, ol] for ol in oils)
        >= min_hardness * dv_prod[mth]
    )
    model.Add(
        model.Sum(hardness[ol] * dv_oil_consume[mth, ol] for ol in oils)
        <= max_hardness * dv_prod[mth]
    )

# 6. Total Tons of oil consumed every month should be equal to the
#    Tons of the food produced in that month.
for mth in months:
    model.Add(model.Sum(dv_oil_consume[mth, ol] for ol in oils) == dv_prod[mth])

# =============================================================================
# # Objective function
# =============================================================================
obj_func = (
    model.Sum(dv_prod[mth] * price for mth in months)
    - model.Sum(cost[mth_ol] * dv_oil_buy[mth_ol] for mth_ol in month_oil)
    - model.Sum(
        inventory_holding_cost * dv_oil_inventory[mth_ol] for mth_ol in month_oil
    )
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
