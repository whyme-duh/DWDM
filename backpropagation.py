import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(0)
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, size=100)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

class VisualizationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
            xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) > 0.5
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
            plt.title('Epoch: {:d}'.format(epoch))
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()

history = model.fit(X, y, epochs=50, callbacks=[VisualizationCallback()])

plt.figure()
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
