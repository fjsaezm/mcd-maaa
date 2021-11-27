import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

def createDataSet(n,model,ymargin,noise=None,output_boundary=False):
    """
    Para crear los problemas, siempre con dos clases y en dos dimensiones. Sus argumentos son:
        - *n*, número de patrones en el problema
        - *model*, tipo de modelo para la frontera que separa las clases, puede ser 'linear', 'square' o 'sine'
        - *ymargin*, margen de separación entre las dos clases, cuanto mayor es *ymargin* más separadas están las clases, valores negativos implican solape entre las clases
        - *noise*, introduce un ruido gausiano a la x e y
        - *output_boundary*, Si vale True la función devuelve la frontera de decisión
    """
    x = np.random.rand(n,1)*2.0*np.pi
    xbnd = np.linspace(0,2.0*np.pi,100)

    if model == 'sine':
        y = (np.random.rand(n,1) - 0.5)*2.2
        c = y > np.sin(x)
        ybnd = np.sin(xbnd)
    elif model == 'linear':
        y = np.random.rand(n,1)*2.0*np.pi
        c = y > x
        ybnd = xbnd
    elif model == 'square':
        y = np.random.rand(n,1)*4.0*np.pi*np.pi
        c = y > x*x
        ybnd = xbnd*xbnd
    else:
        y = np.random.rand(n,1)*2.0*np.pi
        c = y > x
        ybnd = xbnd
    
    y[c == True] = y[c == True] + ymargin
    y[c == False] = y[c == False] - ymargin
    
    if noise is not None:
        y = y + noise * np.random.randn(n,1)
        x = x + noise * np.random.randn(n,1)

    if output_boundary == True:
        return x, y, (c*1).ravel(), xbnd, ybnd
    else:
        return x, y, (c*1).ravel()

def plotModel(x,y,clase,clf,title=""):
    """
    La función *plotModel* la usaremos para dibujar el resultado de un clasificador sobre el conjunto de datos. Sus argumentos son:
        - *x*, coordenada x de los puntos
        - *y*, coordenada y de los puntos
        - *c*, clase de los puntos, si se pasa None, entonces considera que x e y son la frontera real de decisión y la muestra con plot
        - *clf*, el clasificador
        - *title*, título para el gráfico
    """
    from matplotlib.colors import ListedColormap
    
    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2
    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    elif hasattr(clf, "predict_proba"):
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    z = z.reshape(xx.shape)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
    plt.contour(xx, yy, z, [0.5], linewidths=[2], colors=['k'])

    if clase is not None:
        plt.scatter(x[clase==-1], y[clase==-1], c='#FF0000')
        plt.scatter(x[clase==1], y[clase==1], c='#0000FF')
    else:
        plt.plot(x,y,'g', linewidth=3)
        
    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)

def plotData(x,y,c,style0,style1,title=''):
    """
    La función, *plotData*, la usaremos para dibujar los datos. Sus argumentos son:
        - *x*, coordenada x de los puntos
        - *y*, coordenada y de los puntos
        - *c*, clase de los puntos
        - *style0*, estilo con el que pintamos los puntos de la clase 0
        - *style1*, estilo con el que pintamos los puntos de la clase 1
        - *title*, título para el gráfico
    """
    plt.scatter(x[c==-1],y[c==-1],**style0)
    plt.scatter(x[c==1],y[c==1],**style1)
    plt.grid(True)
    plt.axis([x.min()-0.2, x.max()+0.2, y.min()-0.2, y.max()+0.2])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)


########### IMPLEMENTACION DE BAGGING (RANDOM FOREST)

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from scipy.stats import mode

class BaggingCasero:
    def __init__(self, n_estimators=101):
        self.n_estimators = n_estimators
        self._estimators = []
    
    def fit(self,X,y):
        N = X.shape[0]
        for i in range(self.n_estimators):
            """
              Rellenar esta parte para que cree un árbol usando
              una muestra bootstrap de los datos
            """
            X_boostrap, y_boostrap = resample(X, y, replace=True, n_samples=len(y))
            tree = DecisionTreeClassifier()
            tree.fit(X, y)
            self._estimators.append(tree)
            
    def predict(self,X):
        votos = np.zeros((X.shape[0],len(self._estimators)))
        # Calcula la salida de cada árbol para cada dato
        for ie, estimator in enumerate(self._estimators):
            votos[:,ie] = estimator.predict(X)
            
        """
           Calcula la clase más votada de cada ejemplo, es decir,
           la moda
        """
        moda = mode(votos, axis=1)[0]
        return moda
            
#bagging = BaggingCasero()
#bagging.fit(Xtrain, ytrain)
#plotModel(x1 ,x2, ytrain, bagging)