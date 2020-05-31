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

def windowed_multi_combination(evidence, max_number_of_evidences=None, all_data=None, weights=None):
    """
    Windows the evidence.  Only allows the maximum amount (the latest evidences)
    Note: has no effect on this function since the evidence is not retained
    :param evidence: dict of new evidence to add
    :param max_number_of_evidences: the max number of evidences to window
    :param all_data: the data to combine with
    :param weights: dict of weights associated with the new evidence
    """
    if (all_data is not None) and ("number_of_evidences" in all_data) and (max_number_of_evidences is not None) and\
            (max_number_of_evidences > 1):
        all_data["number_of_evidences"] = min(all_data["number_of_evidences"], max_number_of_evidences - len(evidence))
    return multi_combination(evidence, all_data, weights)


def dataset_combination(all_data_1, all_data_2, max_number_of_evidences=None):
    # Yager is same as D-S for this - only has the combined data since prior evidence isn't kept,
    #  so simply combine the two datasets as evidence.
    evidence = {
        "evidence_1": all_data_1,
        "evidence_2": all_data_2
    }
    return windowed_multi_combination(evidence, max_number_of_evidences)


# Combine multiple inputs via Yager's combination rule
# Note: Yager needs to know the universal set.  Since that is not
# explicitly provided, make sure it is in all available inputs
def multi_combination(evidence, all_data=None, weights=None):
    # Weights do not affect Yager.  All inputs assumed to be of equal weight.
    # Create the return
    result = {}

    # First, combine evidence and all_data to create a full set of input data
    inputs = {}
    for evidence_key in evidence.keys():
        inputs[evidence_key] = evidence[evidence_key]
    if (all_data is not None) and all_data:
        inputs["all_data_key"] = all_data

    # Loop and combine
    first = True
    for sensor_key in inputs.keys():
        if first is True:
            for input_key in inputs[sensor_key].keys():
                if isinstance(input_key, tuple) is True:
                    store_key = tuple(sorted(input_key))
                else:
                    store_key = (input_key,)
                result[store_key] = inputs[sensor_key][input_key]
        else:
            second_input = {}
            for input_key in inputs[sensor_key].keys():
                if isinstance(input_key, tuple) is True:
                    store_key = tuple(sorted(input_key))
                else:
                    store_key = (input_key,)
                second_input[store_key] = inputs[sensor_key][input_key]
            if second_input:
                result = combination(result, second_input)
        first = False
    return result


# Implements Yager's combination rule
# The universal set must be as an input in either dic1 or dic2
def combination(dic1, dic2):
    # Extract the sets
    sets = set(dic1.keys()).union(set(dic2.keys()))
    result = dict.fromkeys(sets, 0)

    # Combination process
    for i in dic1.keys():
        for j in dic2.keys():
            if set(i).intersection(set(j)) == set(i):
                result[i] += dic1[i] * dic2[j]
            elif set(i).intersection(set(j)) == set(j):
                result[j] += dic1[i] * dic2[j]

    # Allocate the unallocated belief mass to the universal set (to the unknown)
    f = sum(list(result.values()))
    f = 1 - f
    if f > 0:
        # The universal set will always have the most values in the
        #  tuple key
        max_len = 0
        universal_set = None
        for output_key in result.keys():
            if len(output_key) > max_len:
                max_len = len(output_key)
                universal_set = output_key
        if universal_set is not None:
            result[universal_set] += f
    return result


def final_probabilities(all_data):
    """
    For a consistent interface with ECR
    :param all_data: The data of all information based on this combination method
    :return: The dictionary of probabilities for all options.  Should be the same as all_data for this method
    """
    return all_data
