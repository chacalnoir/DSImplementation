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
from combinationRules.utilities import the_keys
from functools import reduce
from math import sqrt
from copy import deepcopy


def windowed_multi_combination(evidence, max_number_of_evidences=None, all_data=None, weights=None):
    """
    Windows the evidence.  Only allows the maximum amount (the latest evidences)
    :param evidence: dict of new evidence to add
    :param max_number_of_evidences: the max number of evidences to window
    :param all_data: the data to combine with
    :param weights: dict of weights associated with the new evidence
    """
    if (all_data is not None) and ("number_of_evidences" in all_data) and (max_number_of_evidences is not None) and\
            (max_number_of_evidences > 1):
        # May need to limit the data
        if (len(evidence) + all_data["number_of_evidences"]) > max_number_of_evidences:
            # Need to limit the data
            num_to_retain = max_number_of_evidences - len(evidence)
            reduce_by = all_data["number_of_evidences"] - num_to_retain
            for counter in range(reduce_by, all_data["number_of_evidences"]):
                all_data["evidence"][counter - reduce_by] = deepcopy(all_data["evidence"][counter])
                all_data["evidence_weights"][counter - reduce_by] = all_data["evidence_weights"][counter]
            all_data["number_of_evidences"] = num_to_retain
            while len(all_data["evidence"]) > num_to_retain:
                # Always pops the one just beyond the correct ones to keep to reduce the size.
                all_data["evidence"].pop(num_to_retain, None)
            # combined and last_evidence will be reset in the next call
    # Combine the evidence
    return multi_combination(evidence, all_data, weights)


def dataset_combination(all_data_1, all_data_2, max_number_of_evidences=None):
    # Take the evidence and add it to the first dataset
    return windowed_multi_combination(all_data_2["evidence"], max_number_of_evidences,
                                      all_data_2["evidence_weights"], all_data_1)


# Combine multiple inputs via Zhang's combination rule
def multi_combination(evidence, all_data=None, weights=None):
    # Create the return if necessary
    if all_data is None:
        all_data = {
            "evidence": {},
            "evidence_weights": {},
            "number_of_evidences": 0,
            "combined": {},
            "last_evidence": {}  # For plotting with the last update visible
        }
    else:
        # Make sure all data is appropriately set
        if "evidence" not in all_data:
            all_data["evidence"] = {}
        if "evidence_weights" not in all_data:
            all_data["evidence_weights"] = {}
        if "number_of_evidences" not in all_data:
            if not all_data["evidence"]:
                all_data["number_of_evidences"] = 0
            else:
                all_data["number_of_evidences"] = max(all_data["evidence"].keys()) + 1
        if "combined" not in all_data:
            all_data["combined"] = {}
        if "last_evidence" not in all_data:
            all_data["last_evidence"] = {}

    # First, add the evidence into the stored evidence, renumbering the evidence to keep the keys unique
    for evidence_key in evidence.keys():
        all_data["last_evidence"] = {}  # Reset
        all_data["evidence"][all_data["number_of_evidences"]] = deepcopy(evidence[evidence_key])
        for input_key in all_data["evidence"][all_data["number_of_evidences"]]:
            # Convert any keys that aren't tuples to tuples
            if isinstance(input_key, tuple) is False:
                # Convert to a tuple
                new_key = (input_key,)
            else:
                # Sort the tuple to make sure everything aligns properly
                new_key = tuple(sorted(input_key))
            store_value = all_data["evidence"][all_data["number_of_evidences"]][input_key]
            all_data["evidence"][all_data["number_of_evidences"]].pop(input_key, None)
            all_data["evidence"][all_data["number_of_evidences"]][new_key] = store_value
            # Save for ease of access later
            all_data["last_evidence"][new_key] = store_value
        if (weights is not None) and (evidence_key in weights):
            all_data["evidence_weights"][all_data["number_of_evidences"]] = weights[evidence_key]
        else:
            all_data["evidence_weights"][all_data["number_of_evidences"]] = 1.0
        all_data["number_of_evidences"] += 1

    # Create required dictionaries
    pignist_vector = {}
    cos_dict = {}
    sup_dict = {}
    crd_dict = {}
    mae_dict = {}

    # List the full set of inputs
    thetas = list(reduce(lambda a, b: a | set(the_keys(b.keys())), all_data["evidence"].values(), set()))
    # the global ignorance set might be in here
    powerset = [tuple(sorted([x for j, x in enumerate(thetas) if (i >> j) & 1])) for i in range(2 ** len(thetas))]
    # remove the null set from the powerset
    powerset.remove(())

    # Loop through the inputs
    for sensor in all_data["evidence"].keys():
        # Count through each sensor
        pignist_vector[sensor] = []  # Create

        # Create the pignist vectors
        # n-dimension should be 3 (a, b, c)
        for single_input in thetas:
            value = 0.0
            for set_input in powerset:
                # Get null belief (open world case)
                if "zero" in all_data["evidence"][sensor]:
                    null_set = all_data["evidence"][sensor]["zero"]
                else:
                    null_set = 0.0

                if (set_input in all_data["evidence"][sensor]) and \
                        (single_input in set_input) and (all_data["evidence"][sensor][set_input] > 0.0):
                    value += (1 / len(set_input)) * (all_data["evidence"][sensor][set_input] / (1 - null_set))
                    # else: zero value - doesn't add in
            # Append to the pignist vector
            pignist_vector[sensor].append(value)

    # Calculate the conflict/angle between the evidences
    for i in all_data["evidence"].keys():
        # Initialize
        cos_dict[i] = {}
        for j in all_data["evidence"].keys():
            # Calculate the angle between
            if i == j:
                # Cos of the same keys is always 1 (along the diagonal) - this speed it up slightly and reduces
                #  roundoff error.
                cos_dict[i][j] = 1.0
            else:
                dot = 0.0
                length_i = 0.0
                length_j = 0.0
                for index in range(0, len(pignist_vector[i])):
                    dot += pignist_vector[i][index] * pignist_vector[j][index]
                    length_i += pow(pignist_vector[i][index], 2)
                    length_j += pow(pignist_vector[j][index], 2)
                # Finish the Euclidean length computations
                length_i = sqrt(length_i)
                length_j = sqrt(length_j)
                # Get the cosine of the angle between the vectors
                cos_dict[i][j] = dot / (length_i * length_j)

    # Calculate the degree of support for evidence i
    #  and calculate the degree of support for all evidences, prepared for normalization
    sum_sup = 0.0
    for i in all_data["evidence"].keys():
        # Initialize
        sup_dict[i] = 0.0
        for j in all_data["evidence"].keys():
            sup_dict[i] += cos_dict[i][j]

        # Sum for sum_sup
        sum_sup += sup_dict[i]

    # Normalize
    for i in all_data["evidence"].keys():
        crd_dict[i] = sup_dict[i] / sum_sup

    # Calculate the weighted average credibility of the original reliability
    mae_dict_sum = 0.0
    for input_name in powerset:
        mae_dict[input_name] = 0.0
        for i in all_data["evidence"].keys():
            # If not in existence, effectively a zero
            if input_name in all_data["evidence"][i]:
                input_weight = 1.0
                if i in all_data["evidence_weights"]:
                    input_weight = all_data["evidence_weights"][i]
                add_mass = crd_dict[i] * all_data["evidence"][i][input_name] * input_weight
                mae_dict[input_name] += add_mass
                mae_dict_sum += add_mass
                # else: effectively a zero
    # Normalize for weighting
    for input_name in powerset:
        mae_dict[input_name] /= mae_dict_sum

    # Set up for the DS combination loop
    second_input = {}
    for input_key in mae_dict.keys():
        all_data["combined"][input_key] = mae_dict[input_key]
        second_input[input_key] = all_data["combined"][input_key]

    # Combine with Dempster-Shafer using the reformed mass as the input for all sensors
    for sensor in range(2, len(all_data["evidence"].keys()) + 1):
        all_data["combined"] = combination(all_data["combined"], second_input)

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
