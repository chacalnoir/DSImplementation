# --------------------------------------------------------------------------
# Copyright 2020 Joel Dunham

# This file is part of DSImplementation.

# DSImplementation is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# DSImplementation is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DSImplementation.  If not, see <https://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------

from combinationRules.dsCombination import combination
from copy import deepcopy

ROUNDOFF_DELTA = 1e-4


def windowed_multi_combination(evidence, max_number_of_evidences=None, all_data=None, weights=None):
    """
    Pseudo-windows the evidence.  Only allows the maximum amount (the latest evidences).  Does not remove old evidence,
     but rather limits the number of combinations, thereby acting as though the old evidence is historical evidence.
     Because it doesn't remove old evidence, it always combines up to max since the old evidence has full weight.
    :param evidence: dict of new evidence to add
    :param max_number_of_evidences: the max number of evidences to window
    :param all_data: the data to combine with
    :param weights: dict of weights associated with the new evidence
    """
    if (all_data is not None) and ("number_of_evidences" in all_data) and (max_number_of_evidences is not None) and\
            (max_number_of_evidences > 1):
        all_data["number_of_evidences"] = max(min(all_data["number_of_evidences"],
                                                  max_number_of_evidences - len(evidence)), 0)
    return multi_combination(evidence, all_data, weights)


def dataset_combination(all_data_1, all_data_2, max_number_of_evidences=None):
    """
    Combines the two datasets.  This is fairly easy for Murphy since it's a weighted average combined multiple times.
    """
    each_weight = all_data_2["evidence_weight"] / all_data_2["number_of_evidences"]
    evidence = {}
    weight = {}
    for counter in range(0, all_data_2["number_of_evidences"]):
        evidence[counter] = deepcopy(all_data_2["evidence"])
        weight[counter] = each_weight
    return windowed_multi_combination(evidence, max_number_of_evidences, all_data_1, weight)


# Combine multiple inputs via Murphy's combination rule
# For the purposes of Murphy's rule, evidence and all_data use the same format, just are split for a common
#  interface with ECR
def multi_combination(evidence, all_data=None, weights=None):
    # Create the return if necessary
    if all_data is None:
        all_data = {
            "evidence": {},
            "evidence_weight": 0.0,
            "number_of_evidences": 0,
            "combined": {},
            "last_evidence": {}  # For plotting with the last update visible
        }
    else:
        # Make sure all data is appropriately set
        if "evidence" not in all_data:
            all_data["evidence"] = {}
        if "evidence_weight" not in all_data:
            all_data["evidence_weight"] = 0.0
        if "number_of_evidences" not in all_data:
            all_data["number_of_evidences"] = 0
        if "combined" not in all_data:
            all_data["combined"] = {}
        if "last_evidence" not in all_data:
            all_data["last_evidence"] = {}

    # Combine all evidence into the existing evidence to create the full set of input data
    for evidence_key in evidence.keys():
        # Get the weight for this evidence
        mass_weight = 1.0
        if (weights is not None) and (evidence_key in weights):
            mass_weight = weights[evidence_key]
        # Combine/weighted average each new piece of evidence
        all_keys = list(all_data["evidence"].keys())
        all_data["last_evidence"] = {}  # Reset
        for mass_key, mass_value in evidence[evidence_key].items():
            if isinstance(mass_key, tuple) is True:
                store_key = tuple(sorted(mass_key))
            else:
                store_key = (mass_key,)
            current_evidence = 0.0
            # Weighted average the new data
            if store_key in all_data["evidence"]:
                current_evidence = all_data["evidence"][store_key] * all_data["evidence_weight"]
                all_keys.remove(store_key)
            all_data["evidence"][store_key] = (current_evidence + mass_value * mass_weight) /\
                                              (all_data["evidence_weight"] + mass_weight)
            all_data["last_evidence"][store_key] = mass_value
        # Update the ones that didn't get updated
        for mass_key in all_keys:
            all_data["evidence"][mass_key] = (all_data["evidence"][mass_key] * all_data["evidence_weight"]) / \
                                             (all_data["evidence_weight"] + mass_weight)
        all_data["number_of_evidences"] += 1
        all_data["evidence_weight"] += mass_weight

    # Loop and combine
    # Murphy uses averages, so all have to be combined at the same time
    all_data["combined"] = deepcopy(all_data["evidence"])
    second_input = deepcopy(all_data["evidence"])
    for input_counter in range(1, all_data["number_of_evidences"]):  # One less since starting from 1: correct times
        all_data["combined"] = combination(all_data["combined"], second_input)

    # Return the full internal data
    return all_data


def final_probabilities(all_data):
    """
    For a consistent interface with ECR
    :param all_data: The data of all information based on this combination method
    :return: The dictionary of probabilities for all options.
    """
    if (all_data is not None) and ("combined" in all_data):
        return all_data["combined"]
    else:
        # Return none if no available data
        return None
