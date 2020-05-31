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

from copy import deepcopy

ROUNDOFF_DELTA = 1e-4


def windowed_multi_combination(evidence, max_number_of_evidences=None, all_data=None, weights=None):
    """
    Windows the evidence.  Only allows the maximum amount (the latest evidences)
    Note: has no effect on this function since the evidence is not retained
    :param evidence: dict of new evidence to add
    :param max_number_of_evidences: the max number of evidences to window
    :param all_data: the data to combine with
    :param weights: dict of weights associated with the new evidence
    """
    return multi_combination(evidence, all_data, weights)


def dataset_combination(all_data_1, all_data_2, max_number_of_evidences=None):
    # Take the evidence and add it to the first dataset - will overwrite with last evidence
    return windowed_multi_combination(all_data_2, max_number_of_evidences, all_data_1)


# Overwrite with newest data
def multi_combination(evidence, all_data=None, weights=None):
    # Save the newest data
    all_data = deepcopy(evidence[max(list(evidence.keys()))])

    # Return the full internal data
    return all_data


def final_probabilities(all_data):
    """
    For a consistent interface with ECR
    :param all_data: The data of all information based on this combination method
    :return: The dictionary of probabilities for all options.
    """
    if all_data is not None:
        return all_data
    else:
        # Return none if no available data
        return None
