import numpy as np

class kmeans:
    def __init__(self):
        pass

    def distance(self, X, Y):
        """
        Inputs
        X: array like with dimension M x k 
        Y: array like with dimension N x k
        """
        
        #Euclidean distance
        x_sqrd = np.sum(np.square(X), axis=1)
        y_sqrd = np.sum(np.square(Y), axis=1)

        #Distance matrix
        #See in deep here https://www.dabblingbadger.com/blog/2020/2/27/implementing-euclidean-distance-matrix-calculations-from-scratch-in-python
        X_x_Y = X @ Y.T

        """
        Return
        D: Matrix with dimension of MxN, Each value correspond to the distance
        of the element i in X and the element j in Y
        """
        D = np.sqrt(abs(x_sqrd[:, np.newaxis]+y_sqrd-2*X_x_Y)) #Reshape x_sqrd to obtain the MxN matrix

        return D

    def rand_centroids(self, X, nclusters):
        """
        Inputs
        X: array like whit dimension M x k
        nclusters: number of clusters
        """
        #Dimension of the array
        rows, colms = X.shape
        # Initial mean vector
        mean_vector = np.empty([nclusters,colms])

        #Obtain k random parts of the data
        shuffled_data = np.random.permutation(X)
        partitions    = np.array_split(shuffled_data, nclusters)
        
        #Obtain the centroids, mean values of the shuffle data
        for n, part in enumerate(partitions):
            mean = np.mean(part,axis=0)
            mean_vector[n] = mean
        
        """
        Return
        mean_vector: centroids vector, whit dimension k x D (k: number of clusters)
        """
        return mean_vector

    def assignment_vector(self,X,mean_vector):
        """
        Inputs
        X: array like whit dimension M x D, where D is the dimension
        mean_vector: initial centroids vector, whit dimension k x D (k: number of clusters)
        """
        #Dimension of the array
        rows, colms = X.shape   
        k_index     = np.empty([rows])
        eu_distance = self.distance(X, mean_vector)
        k_index     = np.argmin(eu_distance, axis=1)

        """
        Return
        k_index: index of the minimum distance for each row and assignment for the data
        """
        return k_index

    def update_centroids(self,X,centroids,k_index):
        """
        Inputs
        X: array like whit dimension M x D, where D is the dimension
        centroids: old centroids vector, whit dimension k x D (k: number of clusters)
        k_index: index of the minimum distance for each row and assignment for the data
        """
        #Dimension of the array
        rows, colms   = centroids.shape
        new_centroids = np.empty((rows,colms))
        for i in range(rows):
            new_centroids[i] = np.mean(X[k_index == i], axis = 0)
        """
        Return
        new_centroids: new centroids vector, whit dimension k x D (k: number of clusters)
        """    
        return new_centroids

    def losses(self,X,centroids,k_index):
        """
        Inputs
        X: array like whit dimension M x D, where D is the dimension
        centroids: centroids vector, whit dimension k x D (k: number of clusters)
        k_index: index of the minimum distance for each row and assignment for the data
        """    
        #Obtain euclidian distance 
        eu_distance = self.distance(X,centroids)
        #initial loss value
        loss     = 0.0
        #Data dimension
        rows,colms = X.shape

        for i in range(rows):
            loss = loss + np.square(eu_distance[i][k_index[i]])

        """
        Return
        loss: objective function of Kmeans
        """ 
        return loss

    def fit(self, X, nclusters, max_iter=100, abs_tol=1e-16, rel_tol=1e-16, printable=False):
        """
        Inputs
        X: array like whit dimension M x D, where D is the dimension
        nclusters: number of clusters
        max_iter: maximun number of iterations 
        tol: stop criteria
        printable: similar to verbose, print the results of each iteration
        """
        mean_vector = self.rand_centroids(X, nclusters)

        for iter in range(max_iter):
            k_index     = self.assignment_vector(X,mean_vector)
            mean_vector = self.update_centroids(X,mean_vector,k_index)
            loss        = self.losses(X,mean_vector,k_index)
            K = mean_vector.shape[0]
            if iter:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if printable:
                print('iter %d, loss: %.4f' % (iter, loss))
        """
        Return
        loss: Final value of objective function of Kmeans
        mean_vector: centroids of the clusters
        k_index: assignment vector
        """ 
        return loss, mean_vector, k_index