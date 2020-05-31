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

import unittest
from copy import deepcopy


class TestDS(unittest.TestCase):
    def setUp(self):
        self.max_delta = 0.0001
        self.sensor_data = {
            1: {"a": 0.41,
                "b": 0.29,
                "c": 0.3,
                ("a", "b"): 0.0,
                ("a", "c"): 0.0,
                ("b", "c"): 0.0,
                ("a", "b", "c"): 0.0},
            2: {"a": 0.0,
                "b": 0.9,
                "c": 0.1,
                ("a", "b"): 0.0,
                ("a", "c"): 0.0,
                ("b", "c"): 0.0,
                ("a", "b", "c"): 0.0},
            3: {"a": 0.58,
                "b": 0.07,
                "c": 0.0,
                ("a", "b"): 0.0,
                ("a", "c"): 0.35,
                ("b", "c"): 0.0,
                ("a", "b", "c"): 0.0},
            4: {"a": 0.55,
                "b": 0.1,
                "c": 0.0,
                ("a", "b"): 0.0,
                ("a", "c"): 0.35,
                ("b", "c"): 0.0,
                ("a", "b", "c"): 0.0},
            5: {"a": 0.6,
                "b": 0.1,
                "c": 0.0,
                ("a", "b"): 0.0,
                ("a", "c"): 0.3,
                ("b", "c"): 0.0,
                ("a", "b", "c"): 0.0}
        }

        # Set up the test output data
        self.desired_output = {
            "zhang": {
                2: {("a",): 0.0964, ("b",): 0.8119, ("c",): 0.0917, ("a", "c"): 0.0},
                3: {("a",): 0.568114, ("b",): 0.331892, ("c",): 0.092942, ("a", "c"): 0.00844},
                4: {("a",): 0.91420, ("b",): 0.039475, ("c",): 0.039858, ("a", "c"): 0.00826},
                5: {("a",): 0.98199, ("b",): 0.00339, ("c",): 0.01145, ("a", "c"): 0.00317}
            },
            "dempster": {
                2: {("a",): 0.0, ("b",): 0.8969, ("c",): 0.1031, ("a", "c"): 0.0},
                3: {("a",): 0.0, ("b",): 0.6350, ("c",): 0.3650, ("a", "c"): 0.0},
                4: {("a",): 0.0, ("b",): 0.3321, ("c",): 0.6679, ("a", "c"): 0.0},
                5: {("a",): 0.0, ("b",): 0.1422, ("c",): 0.8578, ("a", "c"): 0.0}
            },
            "yager": {
                2: {("a",): 0.0, ("b",): 0.261, ("c",): 0.03, ("a", "c"): 0.0},
                3: {("a",): 0.4112, ("b",): 0.0679, ("c",): 0.0105, ("a", "c"): 0.2481},
                4: {("a",): 0.6508, ("b",): 0.033013, ("c",): 0.00367, ("a", "c"): 0.178633},
                5: {("a",): 0.77323, ("b",): 0.01669, ("c",): 0.00110, ("a", "c"): 0.09375}
            },
            "murphy": {
                2: {("a",): 0.0964, ("b",): 0.8119, ("c",): 0.0917, ("a", "c"): 0.0},
                3: {("a",): 0.4938, ("b",): 0.4180, ("c",): 0.0792, ("a", "c"): 0.0090},
                4: {("a",): 0.8362, ("b",): 0.1147, ("c",): 0.0410, ("a", "c"): 0.0081},
                5: {("a",): 0.9620, ("b",): 0.0210, ("c",): 0.0138, ("a", "c"): 0.0032}
            }
        }

    def run_sensor_combination(self, sensors, name):
        # Set up the data to run the combinations
        sensor_data_subset = {}
        for sensor_key in range(1, sensors + 1):
            sensor_data_subset[sensor_key] = self.sensor_data[sensor_key]

        # Run combination
        multi_combination = None
        final_probabilities = None
        if name == "dempster":
            from combinationRules.dsCombination import multi_combination
            from combinationRules.dsCombination import final_probabilities
        elif name == "zhang":
            from combinationRules.zhangCombination import multi_combination
            from combinationRules.zhangCombination import final_probabilities
        elif name == "yager":
            from combinationRules.yagerCombination import multi_combination
            from combinationRules.yagerCombination import final_probabilities
        elif name == "murphy":
            from combinationRules.murphyCombination import multi_combination
            from combinationRules.murphyCombination import final_probabilities
        else:
            self.assertTrue(0 == 1, "Unknown combination method {}".format(name))

        internal_results = multi_combination(sensor_data_subset)
        results = final_probabilities(internal_results)

        # Check
        for input_key in self.desired_output[name][sensors].keys():
            self.assertAlmostEqual(self.desired_output[name][sensors][input_key],
                                   results[input_key], delta=self.max_delta,
                                   msg="{} for {} with {} sensors".format(name, input_key, sensors))

    def test_2_dempster_sensors(self):
        self.run_sensor_combination(2, "dempster")

    def test_3_dempster_sensors(self):
        self.run_sensor_combination(3, "dempster")

    def test_4_dempster_sensors(self):
        self.run_sensor_combination(4, "dempster")

    def test_5_dempster_sensors(self):
        self.run_sensor_combination(5, "dempster")

    def test_2_zhang_sensors(self):
        self.run_sensor_combination(2, "zhang")

    def test_3_zhang_sensors(self):
        self.run_sensor_combination(3, "zhang")

    def test_4_zhang_sensors(self):
        self.run_sensor_combination(4, "zhang")

    def test_5_zhang_sensors(self):
        self.run_sensor_combination(5, "zhang")

    def test_2_yager_sensors(self):
        self.run_sensor_combination(2, "yager")

    def test_3_yager_sensors(self):
        self.run_sensor_combination(3, "yager")

    def test_4_yager_sensors(self):
        self.run_sensor_combination(4, "yager")

    def test_5_yager_sensors(self):
        self.run_sensor_combination(5, "yager")

    def test_2_murphy_sensors(self):
        self.run_sensor_combination(2, "murphy")

    def test_3_murphy_sensors(self):
        self.run_sensor_combination(3, "murphy")

    def test_4_murphy_sensors(self):
        self.run_sensor_combination(4, "murphy")

    def test_5_murphy_sensors(self):
        self.run_sensor_combination(5, "murphy")

    def test_2_ds_results(self):
        from combinationRules.dsCombination import multi_combination
        evidence = {
            1: {"red_ball": 0.0,
                "green_ball": 0.0,
                "red_cube": 0.1,
                ("red_ball", "green_ball"): 0.8,
                ("green_ball", "red_cube"): 0.0,
                ("red_ball", "red_cube"): 0.0,
                ("red_ball", "green_ball", "red_cube"): 0.1},
            2: {"red_ball": 0.0,
                "green_ball": 0.2,
                "red_cube": 0.0,
                ("red_ball", "green_ball"): 0.0,
                ("green_ball", "red_cube"): 0.0,
                ("red_ball", "red_cube"): 0.6,
                ("red_ball", "green_ball", "red_cube"): 0.2},
        }
        non_weighted_results = multi_combination(evidence)
        test = 1

    def test_paper_calculations(self):
        from combinationRules.dsCombination import multi_combination
        evidence = {
            1: {"a": 0.2,
                "b": 0.5,
                ("a", "b"): 0.3},
            2: {"a": 0.2,
                "b": 0.5,
                ("a", "b"): 0.3},
        }
        non_weighted_results = multi_combination(evidence)

    def test_application_calculations(self):
        from combinationRules.murphyCombination import multi_combination
        evidence = {
            1: {"LOW": 0.0,
                "MEDIUM": 0.0,
                "HIGH": 0.75,
                ("LOW", "MEDIUM"): 0.0,
                ("HIGH", "MEDIUM"): 0.0,
                ("HIGH", "LOW"): 0.0,
                ("HIGH", "LOW", "MEDIUM"): 0.25},
            2: {"LOW": 1.0,
                "MEDIUM": 0.0,
                "HIGH": 0.0,
                ("LOW", "MEDIUM"): 0.0,
                ("HIGH", "MEDIUM"): 0.0,
                ("HIGH", "LOW"): 0.0,
                ("HIGH", "LOW", "MEDIUM"): 0.0},
        }
        results = multi_combination(evidence, weights={1: 3.47883164513351, 2: 1.15643610986581})
        test = 0

    def test_application_calculations_for_average(self):
        from combinationRules.murphyCombination import multi_combination
        evidence = {
            1: {"LOW": 0.7,
                "MEDIUM": 0.0,
                "HIGH": 0.0,
                ("LOW", "MEDIUM"): 0.1,
                ("HIGH", "MEDIUM"): 0.0,
                ("HIGH", "LOW"): 0.0,
                ("HIGH", "LOW", "MEDIUM"): 0.2},
            2: {"LOW": 0.7,
                "MEDIUM": 0.0,
                "HIGH": 0.0,
                ("LOW", "MEDIUM"): 0.1,
                ("HIGH", "MEDIUM"): 0.0,
                ("HIGH", "LOW"): 0.0,
                ("HIGH", "LOW", "MEDIUM"): 0.2},
        }
        results = multi_combination(evidence, weights={1: 0.1, 2: 5.0})
        test = 0

    def test_application_calculations_with_ds(self):
        from combinationRules.dsCombination import multi_combination
        evidence = {
            1: {"LOW": 0.0,
                "MEDIUM": 0.0,
                "HIGH": 0.7,
                ("LOW", "MEDIUM"): 0.0,
                ("HIGH", "MEDIUM"): 0.1,
                ("HIGH", "LOW"): 0.0,
                ("HIGH", "LOW", "MEDIUM"): 0.2},
            2: {"LOW": 0.7,
                "MEDIUM": 0.0,
                "HIGH": 0.0,
                ("LOW", "MEDIUM"): 0.1,
                ("HIGH", "MEDIUM"): 0.0,
                ("HIGH", "LOW"): 0.0,
                ("HIGH", "LOW", "MEDIUM"): 0.2},
        }
        results = multi_combination(evidence, weights={1: 3.47883164513351, 2: 0.15643610986581})
        # Note: Weights have no effect on Dempster's rule.  The only way to implement weights is to use Shafer's
        #  discounting, which then effectively results in Yager's rule, which doesn't work with the reverse
        #  multi-parent solver.
        test = 0

    def test_normalization_with_ds(self):
        from combinationRules.dsCombination import multi_combination
        evidence = {
            1: {('c',): 0.004182517220232905,
                ('a', 'c'): 0.7181052796955626,
                ('b', ): 0.17264142227467427,
                ('a', 'b'): 0.0075129011907618,
                ('a', ): 0.0032756406763107997,
                ('b', 'c'): 0.05621283952987573,
                ('a', 'b', 'c'): 0.03806939941258188},
            2: {('c',): 0.004182517220232905,
                ('a', 'c'): 0.7181052796955626,
                ('b', ): 0.17264142227467427,
                ('a', 'b'): 0.0075129011907618,
                ('a', ): 0.0032756406763107997,
                ('b', 'c'): 0.05621283952987573,
                ('a', 'b', 'c'): 0.03806939941258188},
        }
        results = multi_combination(evidence)
        for each_key in evidence[2].keys():
            evidence[2][each_key] *= 1.5
        unscaled_high_results = multi_combination(evidence)
        for each_key in evidence[2].keys():
            evidence[2][each_key] *= 0.5
        unscaled_low_results = multi_combination(evidence)
        for marginal_key, marginal_value in results.items():
            self.assertAlmostEqual(marginal_value, unscaled_high_results[marginal_key], delta=0.0001,
                                   msg="Key {} failed for original to scaled high".format(marginal_key))
            self.assertAlmostEqual(marginal_value, unscaled_low_results[marginal_key], delta=0.0001,
                                   msg="Key {} failed for original to scaled low".format(marginal_key))

    def test_multi_step_murphy(self):
        """
        Tests ability for Murphy to add new information versus all-at-once information
        """
        # Test without weights
        old_results = None
        new_results = None
        delta_method_1_results = None
        delta_method_2_results = None
        for counter in range(1, 6):
            evidence = {
                counter: self.sensor_data[counter]
            }

            # New, correct update method
            from combinationRules.murphyCombination import multi_combination
            new_results = multi_combination(evidence, new_results)
            new_probabilities = new_results["combined"]

            # Attempts to get correct results with deltas (not full information)
            evidence_to_enter = None
            from combinationRules.murphyCombination import multi_combination
            for evidence_counter in range(1, counter + 1):
                evidence_to_enter = multi_combination(evidence, evidence_to_enter)
            evidence_to_enter_marginals = {"evidence": evidence_to_enter["combined"]}
            # Method 1: Average the results with the previous results
            if counter == 1:
                delta_method_1_results = deepcopy(evidence_to_enter_marginals["evidence"])
            else:
                # Average
                for stored_key, stored_mass in delta_method_1_results.items():
                    delta_method_1_results[stored_key] = (stored_mass * (counter - 1) +
                                                          evidence_to_enter_marginals["evidence"][stored_key]) /\
                                                         counter
            delta_method_1_probabilities = deepcopy(delta_method_1_results)

            # Method 2: Combine the results with the previous results
            delta_method_2_results = multi_combination(evidence_to_enter_marginals, delta_method_2_results)
            delta_method_2_probabilities = delta_method_2_results["combined"]

            test = 0
