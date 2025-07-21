import tensorflow as tf
from sklearn.metrics import recall_score
import numpy as np

class RecallMetric(tf.keras.metrics.Metric):
    def __init__(self, name='recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_recall = self.add_weight(name="total_recall", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binary prediction

        # Safely compute recall using sklearn inside tf.py_function
        def safe_recall(yt, yp):
            try:
                # Convert to numpy
                yt_np = yt.numpy().astype(int)
                yp_np = yp.numpy().astype(int)
                # Compute recall
                from sklearn.metrics import recall_score
                return recall_score(yt_np, yp_np, zero_division=0)
            except Exception as e:
                print(f"Error in recall calculation: {e}")
                return 0.0  # fallback

        recall_value = tf.py_function(
            safe_recall,
            inp=[y_true, y_pred],
            Tout=tf.float32
        )

        # Accumulate
        self.total_recall.assign_add(recall_value)
        self.count.assign_add(1.0)

    def result(self):
        # Avoid division by zero
        return tf.cond(
            self.count > 0,
            lambda: self.total_recall / self.count,
            lambda: 0.0
        )

    def reset_states(self):
        self.total_recall.assign(0.0)
        self.count.assign(0.0)
