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

__version__ = '0.2'

from copy import deepcopy

ZERO_WEIGHT_DELTA = 1e-4
ALL_DATA_WEIGHTING_KEY = "all_data_weighting_"

# Enumeration of combination methods defined here
COMBINATION_METHODS = {
    "DEMPSTER_SHAFER": "DEMPSTER_SHAFER",
    "MURPHY": "MURPHY",
    "YAGER": "YAGER",
    "ZHANG": "ZHANG",
    "OVERWRITE": "OVERWRITE"  # No combination, just overwrite the data
}


def import_and_calculate_probabilities(method, all_data):
    """
    Imports the correct method and returns the probabilities (may be the same or different than all_data
    depending on the method)
    :param method: str: the method in COMBINATION_METHODS
    :param all_data: dict: the internal data of the method
    :return: the probabilities for that data
    """
    if method == COMBINATION_METHODS["DEMPSTER_SHAFER"]:
        from combinationRules.dsCombination import final_probabilities
        probabilities = final_probabilities(all_data)
    elif method == COMBINATION_METHODS["MURPHY"]:
        from combinationRules.murphyCombination import final_probabilities
        probabilities = final_probabilities(all_data)
    elif method == COMBINATION_METHODS["YAGER"]:
        from combinationRules.yagerCombination import final_probabilities
        probabilities = final_probabilities(all_data)
    elif method == COMBINATION_METHODS["ZHANG"]:
        from combinationRules.zhangCombination import final_probabilities
        probabilities = final_probabilities(all_data)
    elif method == COMBINATION_METHODS["OVERWRITE"]:
        from combinationRules.overwrite import final_probabilities
        probabilities = final_probabilities(all_data)
    elif method is None:
        raise ValueError("import_and_combine: None type method - cannot calculate probabilities")
    else:
        raise ValueError("import_and_combine: unknown method type " + method)

    if probabilities is not None:
        for marginal_key, marginal_value in probabilities.items():
            # Rounding can cause issues
            probabilities[marginal_key] = min(max(marginal_value, 0.0), 1.0)
    return probabilities


def import_and_combine_datasets(method, all_data_1, all_data_2, max_number_of_evidences=None):
    """
    Imports the correct method, combines the two datasets, and returns the result
    Note: since Python uses references and changes the dictionaries, deepcopies are required to avoid screwing up
     the internal data
    :param method: dict: The method in COMBINATION_METHODS
    :param all_data_1: dict: the first previous data
    :param all_data_2: dict: the second previous data
    :param max_number_of_evidences: None if no window, > 1 if a window is defined
    :return: dict: the resulting data
    """
    if method == COMBINATION_METHODS["DEMPSTER_SHAFER"]:
        from combinationRules.dsCombination import dataset_combination
        return dataset_combination(deepcopy(all_data_1), deepcopy(all_data_2), max_number_of_evidences)
    elif method == COMBINATION_METHODS["MURPHY"]:
        from combinationRules.murphyCombination import dataset_combination
        return dataset_combination(deepcopy(all_data_1), deepcopy(all_data_2), max_number_of_evidences)
    elif method == COMBINATION_METHODS["YAGER"]:
        from combinationRules.yagerCombination import dataset_combination
        return dataset_combination(deepcopy(all_data_1), deepcopy(all_data_2), max_number_of_evidences)
    elif method == COMBINATION_METHODS["ZHANG"]:
        from combinationRules.zhangCombination import dataset_combination
        return dataset_combination(deepcopy(all_data_1), deepcopy(all_data_2), max_number_of_evidences)
    elif method == COMBINATION_METHODS["OVERWRITE"]:
        from combinationRules.overwrite import dataset_combination
        return dataset_combination(deepcopy(all_data_1), deepcopy(all_data_2), max_number_of_evidences)
    elif method is None:
        raise ValueError("import_and_combine: None type method - cannot combine")
    else:
        raise ValueError("import_and_combine: unknown method type " + method)


def import_and_combine(method, evidence, all_data=None, input_weight=0.0, use_all_data_weight=True):
    """
    Imports the correct method, combines the data, and returns the result
    :param method: dict: The method in COMBINATION_METHODS
    :param evidence: dict: the new evidence
    :param all_data: dict: previous data
    :param input_weight: float weight of the input data relative to the all_data weight - 0.0 for no weighting
    :param use_all_data_weight: boolean whether to use the stored all_data weight or consider that to be 1.0
    :return: dict: the resulting data
    """
    weights = None
    if input_weight > ZERO_WEIGHT_DELTA:
        weights = {}
        for evidence_key in evidence.keys():
            weights[evidence_key] = input_weight
    # Individual methods determine how to handle weights

    if method == COMBINATION_METHODS["DEMPSTER_SHAFER"]:
        from combinationRules.dsCombination import multi_combination
        return multi_combination(evidence, all_data, weights=weights)
    elif method == COMBINATION_METHODS["MURPHY"]:
        from combinationRules.murphyCombination import multi_combination
        return multi_combination(evidence, all_data, weights=weights)
    elif method == COMBINATION_METHODS["YAGER"]:
        from combinationRules.yagerCombination import multi_combination
        return multi_combination(evidence, all_data, weights=weights)
    elif method == COMBINATION_METHODS["ZHANG"]:
        from combinationRules.zhangCombination import multi_combination
        return multi_combination(evidence, all_data, weights=weights)
    elif method == COMBINATION_METHODS["OVERWRITE"]:
        from combinationRules.overwrite import multi_combination
        return multi_combination(evidence, all_data, weights=weights)
    elif method is None:
        raise ValueError("import_and_combine: None type method - cannot combine")
    else:
        raise ValueError("import_and_combine: unknown method type " + method)


def import_and_windowed_combine(method, evidence, max_number_of_evidences=None, all_data=None, input_weight=0.0,
                                use_all_data_weight=True):
    """
    Imports the correct method, combines the data, and returns the result.  Windows the data based on the max
     number of evidences
    :param method: dict: The method in COMBINATION_METHODS
    :param evidence: dict: the new evidence
    :param all_data: dict: previous data
    :param input_weight: float weight of the input data relative to the all_data weight - 0.0 for no weighting
    :param use_all_data_weight: boolean whether to use the stored all_data weight or consider that to be 1.0
    :param max_number_of_evidences: the max number of evidences to window
    :return: dict: the resulting data
    """
    weights = None
    if input_weight > ZERO_WEIGHT_DELTA:
        weights = {}
        for evidence_key in evidence.keys():
            weights[evidence_key] = input_weight
    # Individual methods determine how to handle weights

    if method == COMBINATION_METHODS["DEMPSTER_SHAFER"]:
        from combinationRules.dsCombination import windowed_multi_combination
        return windowed_multi_combination(evidence, max_number_of_evidences, all_data, weights=weights)
    elif method == COMBINATION_METHODS["MURPHY"]:
        from combinationRules.murphyCombination import windowed_multi_combination
        return windowed_multi_combination(evidence, max_number_of_evidences, all_data, weights=weights)
    elif method == COMBINATION_METHODS["YAGER"]:
        from combinationRules.yagerCombination import windowed_multi_combination
        return windowed_multi_combination(evidence, max_number_of_evidences, all_data, weights=weights)
    elif method == COMBINATION_METHODS["ZHANG"]:
        from combinationRules.zhangCombination import windowed_multi_combination
        return windowed_multi_combination(evidence, max_number_of_evidences, all_data, weights=weights)
    elif method == COMBINATION_METHODS["OVERWRITE"]:
        from combinationRules.overwrite import windowed_multi_combination
        return windowed_multi_combination(evidence, max_number_of_evidences, all_data, weights=weights)
    elif method is None:
        raise ValueError("import_and_combine: None type method - cannot combine")
    else:
        raise ValueError("import_and_combine: unknown method type " + method)
