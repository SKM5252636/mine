import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Sample in-memory data: (user_id, product_id)
data = [
    (1, 101), (1, 102), (2, 101), (2, 103),
    (3, 104), (3, 105), (4, 102), (4, 106)
]

# Create mappings
user_ids = sorted(set(x[0] for x in data))
product_ids = sorted(set(x[1] for x in data))
user_map = {uid: i for i, uid in enumerate(user_ids)}
prod_map = {pid: i for i, pid in enumerate(product_ids)}
idx_to_product_id = {v: k for k, v in prod_map.items()}

# Prepare data
X = np.array([[user_map[u], prod_map[p]] for u, p in data])
y = np.ones(len(X), dtype=np.float32)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model
class CFModel(tf.keras.Model):
    def __init__(self, num_users, num_items, k):
        super().__init__()
        self.user_emb = tf.keras.layers.Embedding(num_users, k)
        self.item_emb = tf.keras.layers.Embedding(num_items, k)
    def call(self, inputs):
        u, i = inputs
        return tf.reduce_sum(self.user_emb(u) * self.item_emb(i), axis=1)

# Train
model = CFModel(len(user_map), len(prod_map), 10)
model.compile(optimizer='adam', loss='mse')
model.fit([X_train[:, 0], X_train[:, 1]], y_train,
          validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
          epochs=5, batch_size=2)

print("Test Loss:", model.evaluate([X_test[:, 0], X_test[:, 1]], y_test))

# Recommend
def recommend(user_id, top_n=3):
    if user_id not in user_map:
        return []
    u_idx = user_map[user_id]
    u_vec = model.user_emb(tf.constant([u_idx])).numpy()
    i_vecs = model.item_emb(tf.constant(np.arange(len(prod_map)))).numpy()
    scores = np.dot(u_vec, i_vecs.T)[0]
    top_idxs = np.argsort(scores)[::-1][:top_n]
    return [idx_to_product_id[i] for i in top_idxs]

print("Recommended products for User 1:", recommend(1, 3))
