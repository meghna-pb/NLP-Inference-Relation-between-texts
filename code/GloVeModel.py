import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class GloVeModel():
    
    def __init__(self, n_dim, learning_rate, vocab, max_epochs, x_max, alpha):
        vocab_size = len(vocab)
        self.d = n_dim
        self.eta = learning_rate
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.max_epochs = max_epochs
        self.x_max = x_max
        self.alpha = alpha
        
        # Random initialization of word vectors
        self.W = 1e-2 * np.random.randn(vocab_size, n_dim)
        self.W_tilda = 1e-2 * np.random.randn(vocab_size, n_dim)
        self.b = np.random.randn(vocab_size, 1) # bias term
        self.b_tilda = np.random.randn(vocab_size, 1) # bias term

    def fit(self, X):
        for epoch in range(self.max_epochs):
            print(f'Epoch {epoch+1}/{self.max_epochs}')
            
            cx = X.tocoo()
            assert len(cx.row) == len(cx.col) == len(cx.data)
            nnz = len(cx.row) # Number of Non-Zero
            
            # permute the non-zero values before looping
            perm = np.random.permutation(nnz)
            with tqdm(total=nnz, position=0, leave=True) as pbar:
                for i_iter, (i, j, x) in enumerate(zip(cx.row[perm], cx.col[perm], cx.data[perm])):
                    pbar.update(1)
                    
                    # Si la fréquence de co-occurrence x est inférieure à x_max, appliquez une fonction de pondération.
                    # Sinon, utilisez 1 pour éviter de donner trop de poids aux fréquences élevées.
                    if x < self.x_max:
                        f_ij = (x / self.x_max) ** self.alpha
                    else:
                        f_ij = 1

                    # Calcul du facteur commun utilisé dans le calcul des gradients.
                    # Ce facteur comprend la pondération f_ij et la différence entre le produit scalaire des vecteurs de mots et le logarithme de x.
                    common_factor = f_ij * (np.dot(self.W[i, :], self.W_tilda[j, :]) + self.b[i] + self.b_tilda[j] - np.log(x))

                    # Calcul des gradients pour les vecteurs de mots et les termes de biais.
                    grad_w = common_factor * self.W_tilda[j, :]
                    grad_w_tilda = common_factor * self.W[i, :]
                    grad_b = common_factor
                    grad_b_tilda = common_factor

                    # Mise à jour des vecteurs de mots et des termes de biais en soustrayant le produit du gradient et du taux d'apprentissage.
                    self.W[i, :] -= self.eta * grad_w
                    self.W_tilda[j, :] -= self.eta * grad_w_tilda
                    self.b[i] -= self.eta * grad_b
                    self.b_tilda[j] -= self.eta * grad_b_tilda

    def plot_embeddings(self, n_components=2):
        pca = PCA(n_components=n_components)
        W_proj = pca.fit_transform(self.W)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        for i in range(W_proj.shape[0]):
            plt.text(W_proj[i, 0], W_proj[i, 1], self.vocab[i])
        
        ax.set_xlim((W_proj[:, 0].min(), W_proj[:, 0].max()))
        ax.set_ylim((W_proj[:, 1].min(), W_proj[:, 1].max()))
        return fig, ax
