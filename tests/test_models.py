import unittest
import numpy as np
from sklearn.datasets import make_classification
from src.models import (
    get_base_learners,
    get_meta_learner,
    get_stacking_classifier,
    get_mini_neural_network,
    get_stacking_weights_summary
)

class TestModels(unittest.TestCase):
    def test_get_base_learners(self):
        learners = get_base_learners(random_state=42)
        self.assertIsInstance(learners, list)
        names = [name for name, _ in learners]
        self.assertIn('linearsvc', names)
        self.assertIn('rf', names)
        self.assertIn('et', names)
        self.assertIn('dt', names)
        self.assertIn('mlp', names)
        self.assertIn('hgb', names)
        self.assertIn('linearsvcCalibrated', names)

    def test_mini_neural_network(self):
        mlp = get_mini_neural_network(random_state=42)
        self.assertEqual(mlp.hidden_layer_sizes, (64, 32))
        self.assertEqual(mlp.activation, 'relu')
        self.assertEqual(mlp.solver, 'adam')
        
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        mlp.fit(X, y)
        preds = mlp.predict(X)
        self.assertEqual(len(preds), 200)
        self.assertTrue(hasattr(mlp, 'predict_proba'))

    def test_get_meta_learner(self):
        meta = get_meta_learner(random_state=42)
        self.assertTrue(hasattr(meta, 'fit'))
        self.assertTrue(hasattr(meta, 'predict'))
        self.assertEqual(meta.class_weight, 'balanced')

    def test_hist_gradient_boosting(self):
        learners_dict = dict(get_base_learners(random_state=42))
        hgb = learners_dict['hgb']
        X, y = make_classification(n_samples=150, n_features=10, random_state=42)
        hgb.fit(X, y)
        preds = hgb.predict(X)
        self.assertEqual(len(preds), 150)
        self.assertTrue(hasattr(hgb, 'predict_proba'))

    def test_stacking_classifier_profiles(self):
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=42
        )
        
        # Test Default 'edge' Profile
        clf_edge = get_stacking_classifier(random_state=42, profile='edge', cv_splits=3)
        clf_edge.fit(X, y)
        preds = clf_edge.predict(X)
        self.assertEqual(preds.shape, (100,))
        probs = clf_edge.predict_proba(X)
        self.assertEqual(probs.shape, (100, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        # Test weights extraction
        summary = get_stacking_weights_summary(clf_edge)
        self.assertIn('estimators', summary)
        self.assertIn('coefficients', summary)
        self.assertEqual(summary['estimators'], ['linearsvcCalibrated', 'hgb', 'rf'])

        # Test 'balanced' Profile
        clf_balanced = get_stacking_classifier(random_state=42, profile='balanced', cv_splits=3)
        self.assertEqual(len(clf_balanced.estimators), 4)

        # Test 'legacy' Profile
        clf_legacy = get_stacking_classifier(random_state=42, profile='legacy', cv_splits=3)
        self.assertEqual(len(clf_legacy.estimators), 4)

if __name__ == '__main__':
    unittest.main()
