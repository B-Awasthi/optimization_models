#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This model is implementation of manpower planning problem [example #5]
   listed in fifth edition of Modeling Building in Mathematical Programming
   by H. P. Williams on pages 256-257 and 303 – 304
   This example solves a complex staffing problem by creating an optimal
   multi-period operation plan that minimizes the total number of layoffs and costs.
   Optimization model written below uses Google OR-Tools
"""

import itertools

from ortools.linear_solver import pywraplp

# =============================================================================
# Data
# =============================================================================

# Parameters

years = [1, 2, 3]
skills = ["s1", "s2", "s3"]

current_workforce = {"s1": 2000, "s2": 1500, "s3": 1000}
demand = {
    (1, "s1"): 1000,
    (1, "s2"): 1400,
    (1, "s3"): 1000,
    (2, "s1"): 500,
    (2, "s2"): 2000,
    (2, "s3"): 1500,
    (3, "s1"): 0,
    (3, "s2"): 2500,
    (3, "s3"): 2000,
}
new_hire_attrition = {"s1": 0.25, "s2": 0.20, "s3": 0.10}
experienced_attrition = {"s1": 0.10, "s2": 0.05, "s3": 0.05}
downgrade_skill_attrition = 0.50
max_hiring = {
    (1, "s1"): 500,
    (1, "s2"): 800,
    (1, "s3"): 500,
    (2, "s1"): 500,
    (2, "s2"): 800,
    (2, "s3"): 500,
    (3, "s1"): 500,
    (3, "s2"): 800,
    (3, "s3"): 500,
}
max_overmanning = 150
max_parttime = 50
parttime_cap = 0.50
max_train_unskilled = 200
max_train_semiskilled = 0.25

training_cost = {"s1": 400, "s2": 500}
layoff_cost = {"s1": 200, "s2": 500, "s3": 500}
parttime_cost = {"s1": 500, "s2": 400, "s3": 400}
overmanning_cost = {"s1": 1500, "s2": 2000, "s3": 3000}

# model instantiation
model = pywraplp.Solver("manpower_planning", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

# =============================================================================
# # Decision variables:
# =============================================================================

year_skill = list(itertools.product(years, skills))

# 1. Number of workers of each skill to hire in each year
dv_hire = {i: model.NumVar(0, max_hiring[i], "hire_" + str(i)) for i in year_skill}

# 2. Number of part-time workers of each skill working in each year
dv_part_time = {
    i: model.NumVar(0, max_parttime, "part_time_" + str(i)) for i in year_skill
}

# 3. Number of workers of each skill that are available in each year
dv_workforce = {i: model.NumVar(0, 1000000, "workforce_" + str(i)) for i in year_skill}

# 4. Number of workers of each skill that are laid off in each year
dv_layoff = {i: model.NumVar(0, 1000000, "layoff_" + str(i)) for i in year_skill}

# 5. Number of workers of each skill that are overmanned in each year
dv_excess = {i: model.NumVar(0, 1000000, "excess_" + str(i)) for i in year_skill}

year_skill_skill = list(itertools.product(years, skills, skills))

# 6. Number of workers of one skill to retrain to another skill in each year
dv_train = {i: model.NumVar(0, 1000000, "train_" + str(i)) for i in year_skill_skill}

# =============================================================================
# # Constraints:
# =============================================================================

# 1. Initial workforce balance, year == 1:
for s in skills:
    model.Add(
        dv_workforce[1, s]
        == ((1 - experienced_attrition[s]) * current_workforce[s])
        + ((1 - new_hire_attrition[s]) * dv_hire[1, s])
        + model.Sum(
            ((1 - experienced_attrition[s]) * dv_train[1, s2, s]) - dv_train[1, s, s2]
            for s2 in skills
            if s2 < s
        )
        + model.Sum(
            ((1 - downgrade_skill_attrition) * dv_train[1, s2, s]) - dv_train[1, s, s2]
            for s2 in skills
            if s2 > s
        )
        - dv_layoff[1, s]
    )

# 2. Subsequent workforce balance, year > 1:
for t in years[1:]:
    for s in skills:
        model.Add(
            dv_workforce[t, s]
            == ((1 - experienced_attrition[s]) * dv_workforce[t - 1, s])
            + ((1 - new_hire_attrition[s]) * dv_hire[t, s])
            + model.Sum(
                ((1 - experienced_attrition[s]) * dv_train[t, s2, s])
                - dv_train[t, s, s2]
                for s2 in skills
                if s2 < s
            )
            + model.Sum(
                ((1 - downgrade_skill_attrition) * dv_train[t, s2, s])
                - dv_train[t, s, s2]
                for s2 in skills
                if s2 > s
            )
            - dv_layoff[t, s]
        )

# 3. Unskilled training - Unskilled workers trained in an year cannot exceed
#    the maximum allowance. Unskilled workers cannot be immediately transformed
#    into skilled workers.
for t in years:
    model.Add(dv_train[t, "s1", "s2"] <= max_train_unskilled)
    model.Add(dv_train[t, "s1", "s3"] == 0)

# 4. Semi-skilled Training: Semi-skilled workers trained in an year cannot
#    exceed the maximum allowance.
for t in years:
    model.Add(dv_train[t, "s2", "s3"] <= max_train_semiskilled * dv_workforce[t, "s3"])

# 5. Overmanning: Excess workers in year t cannot exceed the maximum allowance.
for t in years:
    model.Add(model.Sum(dv_excess[t, s] for s in skills) <= max_overmanning)

# 6. Demand: Workforce s available in year t equals the required number of
#    workers plus the excess workers and the part-time workers.
for t in years:
    for s in skills:
        model.Add(
            dv_workforce[t, s]
            == demand[t, s] + dv_excess[t, s] + (parttime_cap * dv_part_time[t, s])
        )

# =============================================================================
# Objective function
# =============================================================================

# Objective function 1 :
# Layoffs: Minimize the total layoffs during the planning horizon.
obj_layoffs = model.Sum(dv_layoff[t, s] for t in years for s in skills)

# Objective function 2 :
# Cost: Minimize the total cost (in USD) incurred by training, overmanning,
# part-time workers, and layoffs in the planning horizon.

# obj_cost = model.Sum(
#     (training_cost[s] * dv_train[t, s, skills[skills.index(s) + 1]]) if s < "s3" else 0
#     for t in years
#     for s in skills
# ) + model.Sum(
#     layoff_cost[s] * dv_layoff[t, s]
#     + parttime_cost[s] * dv_part_time[t, s]
#     + overmanning_cost[s] * dv_excess[t, s]
#     for t in years
#     for s in skills
# )


# solving the model

# minimize layoffs
model.Minimize(obj_layoffs)

# minimize cost
# model.Minimize(obj_cost)

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
