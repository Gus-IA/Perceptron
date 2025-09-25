import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from matplotlib import animation, rc
rc('animation', html='html5')

# Función de activación lineal con valores entre -10 y 10
x = np.linspace(-10, 10)
y = x

plt.plot(x, y)
plt.grid(True)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x) = x', fontsize=14)
plt.title('Función de activación lineal', fontsize=14)
plt.plot(x, np.zeros(len(x)), '--k')
plt.show()

# generamos unos datos aleatorios
# y pintamos estos datos
np.random.seed(42)

x = np.random.rand(20)
y = 2*x + (np.random.rand(20)-0.5)*0.5

plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.show()

# usando el perceptrón para pintar la linea
plt.plot(x, y, "b.")
plt.plot(x, 2*x, 'k')
plt.plot(0.5, 2*0.5, 'sr')
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.show()


# ---- clasificación ----

# funciones de pérdida

def mse(y, y_hat):
		return np.mean((y_hat - y)**2)

def bce(y, y_hat):
		return - np.mean(y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat))

# funciones de activación

def linear(x):
		return x

def step(x):
		return x > 0

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Perceptrón

class Perceptron():
	def __init__(self, size, activation, loss):
		self.w = np.random.randn(size)
		self.ws = []
		self.activation = activation
		self.loss = loss

	def __call__(self, w, x):
		return self.activation(np.dot(x, w))

	def fit(self, x, y, epochs, lr):
		x = np.c_[np.ones(len(x)), x]
		for epoch in range(epochs):
				# Batch Gradient Descent
				y_hat = self(self.w, x)
				# función de pérdida
				l = self.loss(y, y_hat)
				# derivadas
				dldh = (y_hat - y)
				dhdw = x
				dldw = np.dot(dldh, dhdw)
				# actualizar pesos
				self.w = self.w - lr*dldw
				# guardar pesos para animación
				self.ws.append(self.w.copy())
				


# cargamos el dataset iris
iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(int) # clsificación binaria

print(X.shape, y.shape)

# mostramos el gráfico
plt.plot(X[y==1, 0], X[y==1, 1], 's', label="Iris Setosa")
plt.plot(X[y==0, 0], X[y==0, 1], 'x', label="No Iris Setosa")
plt.grid()
plt.legend()
plt.xlabel('petal length', fontsize=14)
plt.ylabel('petal width', fontsize=14)
plt.title("Iris dataset", fontsize=14)
plt.show()




np.random.seed(42)

# instanciamos el Perceptron y entrenamos
# usando la función de activación sigmoid, 3 pesos y bce para clasificación binaria
perceptron = Perceptron(3, sigmoid, bce)
epochs, lr = 20, 0.01 # 20 epochs y 0.01 de learning rate
perceptron.fit(X, y, epochs, lr)



# mostramos la animación del perceptron
def plot(epoch, w):
    ax.clear()
    tit = ax.set_title(f"Epoch {epoch+1}", fontsize=14)
    axes = [0, 5, 0, 2]
    x0, x1 = np.meshgrid(
            np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
            np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
        )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    X_new = np.c_[np.ones(len(X_new)), X_new]
    y_predict = perceptron(w, X_new)
    zz = y_predict.reshape(x0.shape)

    ax.plot(X[y==0, 0], X[y==0, 1], "bs", label="Not Iris-Setosa")
    ax.plot(X[y==1, 0], X[y==1, 1], "yo", label="Iris-Setosa")
    custom_cmap = ListedColormap(['#9898ff', '#fafab0'])

    ax.contourf(x0, x1, zz, cmap=custom_cmap)
    ax.set_xlabel("Petal length", fontsize=14)
    ax.set_ylabel("Petal width", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.axis(axes)
    return [tit]

def get_anim(fig, ax, ws):
    def anim(i):
        return plot(i, ws[i])
    return anim

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, autoscale_on=False)
animate = get_anim(fig, ax, perceptron.ws)
anim = animation.FuncAnimation(fig, animate, frames=len(perceptron.ws), interval=100, blit=False)
plt.show()



# nos quedamos con los mejores pesos
w = perceptron.ws[-1]
print(w)


# pasamos unos nuevos datos
x_new = [1, 2, 0.5]
y_pred = perceptron(w, x_new)
print(y_pred) # Iris Setosa



# ---- clasificación multiclase ----


class Perceptron():
	def __init__(self, inputs, outputs, activation, loss, grad_loss):
		inputs = inputs + 1
		self.w = np.random.normal(loc=0.0,
					scale = np.sqrt(2/(inputs+outputs)),
					size = (inputs, outputs))
		self.ws = []
		self.activation = activation
		self.loss = loss
		self.grad_loss = grad_loss

	def __call__(self, w, x):
		return self.activation(np.dot(x, w))

	def fit(self, x, y, epochs, lr, batch_size=None, verbose=True, log_each=1):
		if batch_size == None:
				batch_size = len(x)
		x = np.c_[np.ones(len(x)), x]
		batches = len(x) // batch_size
		for epoch in range(1,epochs+1):
				# Mini-Batch Gradient Descent
				for b in range(batches):
						_x = x[b*batch_size:(b+1)*batch_size]
						_y = y[b*batch_size:(b+1)*batch_size]
						y_hat = self(self.w, _x)
						#print(y_hat.shape)
						# función de pérdida
						l = self.loss(_y, y_hat)
						# derivadas
						dldh = self.grad_loss(_y, y_hat)
						dhdw = _x
						dldw = np.dot(dhdw.T, dldh)
						# actualizar pesos
						self.w = self.w - lr*dldw
				# guardar pesos para animación
				self.ws.append(self.w.copy())
				# print loss
				if verbose and not epoch % log_each:
						print(f"Epoch {epoch}/{epochs} Loss {l}")

	def predict(self, x):
		x = np.c_[np.ones(len(x)), x]
		return self(self.w, x)


# cargamos el dataset iris, normalizando los datos
iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std # normalización

y = iris.target

X.shape, y.shape

# mostramos el gráfico
plt.plot(X[y==0, 0], X[y==0, 1], 's', label="Iris Setosa")
plt.plot(X[y==1, 0], X[y==1, 1], 'x', label="Iris Versicolor")
plt.plot(X[y==2, 0], X[y==2, 1], 'o', label="Iris Virginica")
plt.grid()
plt.legend()
plt.xlabel('petal length', fontsize=14)
plt.ylabel('petal width', fontsize=14)
plt.title("Iris dataset", fontsize=14)
plt.show()


def softmax(x):
	return np.exp(x) / np.exp(x).sum(axis=-1,keepdims=True)

def crossentropy(y, y_hat):
	logits = y_hat[np.arange(len(y_hat)),y]
	entropy = - logits + np.log(np.sum(np.exp(y_hat),axis=-1))
	return entropy.mean()

# solo si usamos softmax
def grad_crossentropy(y, y_hat):
	answers = np.zeros_like(y_hat)
	answers[np.arange(len(y_hat)),y] = 1
	return (- answers + softmax(y_hat)) / y_hat.shape[0]


class SoftmaxRegression(Perceptron):
	def __init__(self, inputs, outputs):
		# usamos activación lineal porque `crossentropy` ya incluye la softmax
		super().__init__(inputs, outputs, linear, crossentropy, grad_crossentropy)


# clasificador multiclase
perceptron = SoftmaxRegression(2, 3)
epochs, lr = 50, 1
perceptron.fit(X_norm, y, epochs, lr, log_each=10)

# mostramos el gráfico
def plot_multiclass(perceptron, N=4):
    fig, axes = plt.subplots(2, N//2, figsize=(10, 8), squeeze=False)
    for n in range(N):
        if n == 0 or n == N-1: t = n + 1 if n == 0 else len(perceptron.ws)-1
        else: t = int(n*len(perceptron.ws)/(N-1))
        ax = axes[n//(N//2), n%(N//2)]
        resolution=0.02
        ax.set_title(f"Epoch {t}", fontsize=14)
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
        X_new = (np.array([xx1.ravel(), xx2.ravel()]).T - X_mean)/X_std
        X_new = np.c_[np.ones(len(X_new)), X_new]
        w = perceptron.ws[t]
        Z = perceptron(w, X_new)
        Z = np.argmax(softmax(Z), axis=1)
        Z = Z.reshape(xx1.shape)
        ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        ax.set_xlabel('petal length', fontsize=12)
        ax.set_ylabel('petal width', fontsize=12)
        classes = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=classes[cl],
                        edgecolor='black')
        if n == N-1:  # Only add legend to the last plot
            ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()

plot_multiclass(perceptron)



# ---- Perceptrón multicapa ----


# solo si usamos activación lineal
def grad_mse(y, y_hat):
		return y_hat - y.reshape(y_hat.shape)

# función de activación
def relu(x):
	return np.maximum(0, x)

def reluPrime(x):
	return x > 0

class MLP():
	def __init__(self, D_in, H, D_out):
		self.w1, self.b1 = np.random.normal(loc=0.0, scale=np.sqrt(2/(D_in+H)), size=(D_in, H)), np.zeros(H)
		self.w2, self.b2 = np.random.normal(loc=0.0, scale=np.sqrt(2/(H+D_out)), size=(H, D_out)), np.zeros(D_out)
		self.ws = []
		self.loss = mse
		self.grad_loss = grad_mse

	def __call__(self, x):
		self.h_pre = np.dot(x, self.w1) + self.b1
		self.h = relu(self.h_pre)
		y_hat = np.dot(self.h, self.w2) + self.b2
		return y_hat

	def fit(self, X, Y, epochs = 100, lr = 0.001, batch_size=None, verbose=True, log_each=1):
		batch_size = len(X) if batch_size == None else batch_size
		batches = len(X) // batch_size
		l = []
		for e in range(1,epochs+1):
				# Mini-Batch Gradient Descent
				_l = []
				for b in range(batches):
						x = X[b*batch_size:(b+1)*batch_size]
						y = Y[b*batch_size:(b+1)*batch_size]
						y_pred = self(x)
						loss = self.loss(y, y_pred)
						_l.append(loss)
						# Backprop
						dldy = self.grad_loss(y, y_pred)
						grad_w2 = np.dot(self.h.T, dldy)
						grad_b2 = dldy.mean(axis=0)
						dldh = np.dot(dldy, self.w2.T)*reluPrime(self.h_pre)
						grad_w1 = np.dot(x.T, dldh)
						grad_b1 = dldh.mean(axis=0)
						# Update (GD)
						self.w1 = self.w1 - lr * grad_w1
						self.b1 = self.b1 - lr * grad_b1
						self.w2 = self.w2 - lr * grad_w2
						self.b2 = self.b2 - lr * grad_b2
				l.append(np.mean(_l))
				self.ws.append((
						self.w1.copy(),
						self.b1.copy(),
						self.w2.copy(),
						self.b2.copy()
				))
				if verbose and not e % log_each:
						print(f'Epoch: {e}/{epochs}, Loss: {np.mean(l):.5f}')

	def predict(self, ws, x):
		w1, b1, w2, b2 = ws
		h = relu(np.dot(x, w1) + b1)
		y_hat = np.dot(h, w2) + b2
		return y_hat


m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)

# mostramos el gráfico
plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.show()

# hacemos un segudo entrenamiento con diferentes datos
model = MLP(D_in=1, H=5, D_out=1)
epochs, lr = 50, 0.0002
model.fit(x, y, epochs, lr, batch_size=1, log_each=10)

# mostramos el gráfico con estos nuevos datos
N = 3
fig, axs = plt.subplots(N, 1, figsize=(4, 8))
for n in range(N):
	if n == 0 or n == N-1: t = n + 1 if n == 0 else len(model.ws)-1
	else: t = int(n*len(model.ws)/(N-1))
	ax = axs[n]
	ax.plot(x, y, "b.")
	_x_new = np.linspace(-3, 3, 100)
	x_new = _x_new.reshape(len(_x_new),1)
	w = model.ws[t]
	y_pred = model.predict(w, x_new)
	ax.plot(_x_new, y_pred, "-k")
	ax.set_xlabel("$x_1$", fontsize=14)
	ax.set_ylabel("$y$", rotation=0, fontsize=14)
	ax.set_title(f"Iteración: {t}")
	ax.grid(True)
	ax.axis([-3,3,0,10])
plt.tight_layout()
plt.show()