import unittest
import numpy as np
from sklearn.datasets import make_classification
from src.models import get_base_learners, get_meta_learner, get_stacking_classifier

class TestModels(unittest.TestCase):
    def test_get_base_learners(self):
        learners = get_base_learners(random_state=42)
        self.assertIsInstance(learners, list)
        names = [name for name, _ in learners]
        self.assertIn('linearsvc', names)
        self.assertIn('rf', names)
        self.assertIn('et', names)
        self.assertIn('hgb', names)
        self.assertIn('dt', names)

    def test_get_meta_learner(self):
        meta = get_meta_learner(random_state=42)
        self.assertTrue(hasattr(meta, 'fit'))
        self.assertTrue(hasattr(meta, 'predict'))

    def test_stacking_classifier_fit_predict(self):
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=42
        )
        
        stacking_clf = get_stacking_classifier(random_state=42, passthrough=True)
        stacking_clf.fit(X, y)
        
        preds = stacking_clf.predict(X)
        self.assertEqual(preds.shape, (100,))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
        
        if hasattr(stacking_clf, "predict_proba"):
            probs = stacking_clf.predict_proba(X)
            self.assertEqual(probs.shape, (100, 2))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
