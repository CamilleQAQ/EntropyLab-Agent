import unittest

import numpy as np

from entropy_agent.classifiers.bt_lstsvm import (
    BTLSTSVM,
    BinaryLSTSVM,
    BinaryLSTWSVM,
    SequentialBTLSTWSVM,
)


class BinaryLSTSVMTests(unittest.TestCase):
    def test_historical_names_are_identity_aliases(self):
        self.assertIs(BinaryLSTWSVM, BinaryLSTSVM)
        self.assertIs(SequentialBTLSTWSVM, BTLSTSVM)

    def test_linear_planes_match_legacy_normal_equations(self):
        x = np.array([[2., 0.], [3., 1.], [1., 1.], [-2., 0.], [-3., -1.]])
        y = np.array([1, 1, 1, -1, -1])
        model = BinaryLSTSVM(kernel="linear", c1=2., c2=3.).fit(x, y)
        p = np.column_stack((x[y == 1], np.ones(3)))
        q = np.column_stack((x[y == -1], np.ones(2)))
        expected1 = -np.linalg.inv(p.T @ p / 2 + q.T @ q + 1e-4 * np.eye(3)) @ q.T @ np.ones(2)
        expected2 = np.linalg.inv(q.T @ q / 3 + p.T @ p + 1e-4 * np.eye(3)) @ p.T @ np.ones(3)
        np.testing.assert_allclose(model._plane1, expected1, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(model._plane2, expected2, rtol=1e-9, atol=1e-9)

    def test_rbf_kernel_matches_matlab_bandwidth_definition(self):
        x = np.array([[0.], [1.], [3.]])
        y = np.array([1, 1, -1])
        model = BinaryLSTSVM(kernel="rbf", bandwidth=2.).fit(x, y)
        scores = model.scores(np.array([[2.]]))
        kernel_row = np.exp(-np.array([4., 1., 1.]) / 4.)
        design = np.r_[kernel_row, 1.]
        self.assertAlmostEqual(scores.positive_distance[0], abs(design @ model._plane1))
        self.assertAlmostEqual(scores.negative_distance[0], abs(design @ model._plane2))

    def test_unequal_rbf_branches_match_legacy_woodbury_formulas(self):
        x = np.array([[0.], [0.6], [1.5], [2.3], [3.1]])
        for y in (np.array([1, 1, -1, -1, -1]), np.array([1, 1, 1, -1, -1])):
            model = BinaryLSTSVM(c1=2., c2=3., bandwidth=1.4).fit(x, y)
            a, b = x[y == 1], x[y == -1]
            anchors = np.vstack((a, b))

            def design(rows):
                squared = (rows - anchors.T) ** 2
                return np.column_stack((np.exp(-squared / 2.8), np.ones(len(rows))))

            g, h = design(a), design(b)
            eye = np.eye(g.shape[1])
            if len(a) < len(b):
                inverse = np.linalg.inv(h.T @ h + 1e-4 * eye)
                first = -(inverse - inverse @ g.T @ np.linalg.inv(2. * np.eye(len(a)) + g @ inverse @ g.T) @ g @ inverse) @ h.T @ np.ones(len(b))
                second = 3. * (inverse - inverse @ g.T @ np.linalg.inv(np.eye(len(a)) / 3. + g @ inverse @ g.T) @ g @ inverse) @ g.T @ np.ones(len(a))
            else:
                inverse = np.linalg.inv(g.T @ g + 1e-4 * eye)
                first = -2. * (inverse - inverse @ h.T @ np.linalg.inv(np.eye(len(b)) / 2. + h @ inverse @ h.T) @ h @ inverse) @ h.T @ np.ones(len(b))
                second = (inverse - inverse @ h.T @ np.linalg.inv(3. * np.eye(len(b)) + h @ inverse @ h.T) @ h @ inverse) @ g.T @ np.ones(len(a))
            np.testing.assert_allclose(model._plane1, first, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(model._plane2, second, rtol=1e-8, atol=1e-8)

    def test_rejects_nonfinite_and_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BinaryLSTSVM(c1=0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            BinaryLSTSVM().fit(np.array([[np.nan], [1.]]), np.array([1, -1]))


class SequentialBTTests(unittest.TestCase):
    def test_predictions_require_no_test_labels(self):
        x = np.array([[-4.], [-3.], [0.], [1.], [4.], [5.]])
        y = np.array([1, 1, 2, 2, 3, 3])
        model = BTLSTSVM(kernel="linear").fit(x, y)
        result = model.predict(np.array([[-3.5], [0.5], [4.5]]))
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.isin(result, (1, 2, 3)).all())
        self.assertEqual(len(model.models_), 2)

    def test_arbitrary_class_ids_and_feature_width_have_no_dataset_assumptions(self):
        rng = np.random.default_rng(91)
        labels = np.repeat(np.array([-7, 42, 900]), 8)
        centers = np.repeat(np.array([-2.0, 0.0, 2.0]), 8)
        x = rng.normal(centers[:, None], 0.05, size=(24, 13))
        model = BTLSTSVM(kernel="linear").fit(
            x, labels, class_order=(42, -7, 900),
        )
        prediction = model.predict(x)
        self.assertEqual(prediction.shape, (24,))
        self.assertTrue(np.isin(prediction, (-7, 42, 900)).all())
        self.assertEqual(model._width, 13)

if __name__ == "__main__":
    unittest.main()
