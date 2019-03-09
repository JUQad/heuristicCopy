#ifndef GUARD_budgetMaintenance_h
#define GUARD_budgetMaintenance_h

#include "loadData.h"
#include "kernel.h"
#include <tuple>
#include <vector>


using LookupTable = std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>;
using Heuristic = std::tuple<INDEX, INDEX>(std::vector<double>&, sparseData&, Kernel const&, LookupTable const&, double);

Heuristic mergeHeuristicWD;
Heuristic mergeHeuristicRandom;
Heuristic mergeHeuristicKernel;
Heuristic mergeHeuristicRandomWD;
Heuristic mergeHeuristicMinAlpha;
Heuristic mergeHeuristicMintwoAlphas;
Heuristic mergeHeuristic59plusWD;

using HeuristicWithmoreVectors = std::tuple<INDEX, INDEX, std::vector<double>>(std::vector<double>&, sparseData&, Kernel const&, LookupTable const&, double, std::vector<double>);

HeuristicWithmoreVectors mergeHeuristicWDVector;

int mergeAndDeleteSVwithmoreVectors(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, HeuristicWithmoreVectors heuristicWithmoreVectors);

int mergeAndDeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, Heuristic heuristic);

double  mergeAndDeleteSV_pVector(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, Heuristic heuristic, std::vector<double>);

int DeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);

std::tuple<int,  double , std::vector<SE>, char > mergeDeleteAdd(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);
#endif
