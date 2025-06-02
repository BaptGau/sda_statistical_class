import unittest
from unittest.mock import patch, Mock
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper

from artefacts.linear_regression.hypothesis_checker import check_linearity, check_residuals_homoscedasticity, \
    check_residuals_normality, check_residuals_autocorrelation, check_no_colinearity, check_all_hypotheses


class TestHypothesisCheckerFunctions(unittest.TestCase):

    def setUp(self):
        self.mock_model = Mock(spec=RegressionResultsWrapper)
        self.mock_model.rsquared = 0.8
        self.mock_model.resid = np.array([0.1, -0.2, 0.05])

        self.X = np.array([[1, 2], [2, 3], [3, 4]])
        self.y = np.array([1, 2, 3])

    def test_check_linearity_true(self):
        self.mock_model.rsquared = 0.85
        self.assertTrue(check_linearity(self.mock_model))

    def test_check_linearity_false(self):
        self.mock_model.rsquared = 0.0
        self.assertFalse(check_linearity(self.mock_model))

    @patch("artefacts.linear_regression.hypothesis_checker.het_goldfeldquandt")
    def test_check_residuals_homoscedasticity_true(self, mock_het_test):
        mock_het_test.return_value = (None, 0.2, None)
        result = check_residuals_homoscedasticity(self.mock_model.resid, self.X)
        self.assertTrue(result)

    @patch("artefacts.linear_regression.hypothesis_checker.het_goldfeldquandt")
    def test_check_residuals_homoscedasticity_false(self, mock_het_test):
        mock_het_test.return_value = (None, 0.01, None)
        result = check_residuals_homoscedasticity(self.mock_model.resid, self.X)
        self.assertFalse(result)

    @patch("artefacts.linear_regression.hypothesis_checker.shapiro")
    def test_check_residuals_normality_shapiro_true(self, mock_shapiro):
        mock_shapiro.return_value = (None, 0.6)
        result = check_residuals_normality(self.mock_model.resid, n_samples=500)
        self.assertTrue(result)

    @patch("artefacts.linear_regression.hypothesis_checker.shapiro")
    def test_check_residuals_normality_shapiro_false(self, mock_shapiro):
        mock_shapiro.return_value = (None, 0.01)
        result = check_residuals_normality(self.mock_model.resid, n_samples=500)
        self.assertFalse(result)

    @patch("artefacts.linear_regression.hypothesis_checker.lilliefors")
    def test_check_residuals_normality_lilliefors_true(self, mock_lilliefors):
        mock_lilliefors.return_value = (None, 0.6)
        result = check_residuals_normality(self.mock_model.resid, n_samples=1500)
        self.assertTrue(result)

    @patch("artefacts.linear_regression.hypothesis_checker.lilliefors")
    def test_check_residuals_normality_lilliefors_false(self, mock_lilliefors):
        mock_lilliefors.return_value = (None, 0.01)
        result = check_residuals_normality(self.mock_model.resid, n_samples=1500)
        self.assertFalse(result)

    @patch("artefacts.linear_regression.hypothesis_checker.durbin_watson")
    def test_check_residuals_autocorrelation_true(self, mock_dw):
        mock_dw.return_value = 2.0
        result = check_residuals_autocorrelation(self.mock_model.resid)
        self.assertTrue(result)

    @patch("artefacts.linear_regression.hypothesis_checker.durbin_watson")
    def test_check_residuals_autocorrelation_false(self, mock_dw):
        mock_dw.return_value = 1.0
        result = check_residuals_autocorrelation(self.mock_model.resid)
        self.assertFalse(result)

    def test_check_no_colinearity_high(self):
        X_high_corr = np.array([[1, 2], [2, 4], [3, 6]])
        self.assertFalse(check_no_colinearity(X_high_corr))

    def test_check_no_colinearity_low(self):
        X_low_corr = np.array([[1, 0], [0, 1], [1, 1]])
        self.assertTrue(check_no_colinearity(X_low_corr))

    def test_check_all_hypotheses_output_keys(self):
        # Make X and y consistent with model.resid
        X = np.random.rand(100, 2)
        y = np.random.rand(100)

        mock_model = Mock(spec=RegressionResultsWrapper)
        mock_model.resid = np.random.normal(0, 1, size=100)
        mock_model.rsquared = 0.5

        results = check_all_hypotheses(X, y, mock_model)
        expected_keys = {
            "linearity",
            "residuals_normality",
            "residuals_homoscedasticity",
            "residuals_no_autocorrelation",
            "features_no_multicolinearity",
        }
        self.assertEqual(set(results.keys()), expected_keys)
