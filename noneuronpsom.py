import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.cluster import KMeans
from collections import Counter
from collections import defaultdict
from scipy.stats import mode
import warnings
import seaborn as sns
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

np.random.seed(0)

import math

def find_best_grid_shape(n_neurons):
    """
    Finds the best rectangular grid shape (rows, cols) for a given number of neurons.
    
    The goal is to make the shape as close to a square as possible.
    
    Parameters:
    - n_neurons (int): Number of neurons in the SOM.
    
    Returns:
    - (rows, cols): Tuple representing the best grid shape.
    """
    best_rows, best_cols = 1, n_neurons  # Start with (1, n) as worst case
    min_aspect_ratio = float("inf")  # To track the best (most square-like) shape

    # Iterate over possible row values (up to sqrt(n_neurons))
    for rows in range(1, int(math.sqrt(n_neurons)) + 1):
        if n_neurons % rows == 0:  # Only consider factors of n_neurons
            cols = n_neurons // rows
            aspect_ratio = max(rows, cols) / min(rows, cols)  # Closer to 1 is better
            if aspect_ratio < min_aspect_ratio:  # Choose the most square-like shape
                min_aspect_ratio = aspect_ratio
                best_rows, best_cols = rows, cols

    # If no good factors were found (i.e., for a prime number), approximate a square
    if best_rows == 1:  
        best_rows = int(math.sqrt(n_neurons))
        best_cols = math.ceil(n_neurons / best_rows)   # Ensure full coverage

    return best_rows, best_cols


def create_random_batches(data, batch_size, seed=None):
        """
        Create random batches of data.
    
        Parameters:
        - data: List or array of data points.
        - batch_size: Size of each batch.
        - seed: Random seed for reproducibility (optional).
    
        Returns:
            - List of random batches.
        """
        if seed is not None:
            np.random.seed(seed)
    
        # Shuffle data
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = [data[i] for i in shuffled_indices]

        # Split into batches
        random_batches = [shuffled_data[i:i + batch_size] for i in range(0, len(shuffled_data), batch_size)]

        return random_batches 

class neuroablatedNPSOM:
    def __init__(self, grid_size, input_dim,lambda_reg = 0.01 ,sigma = 1.0 , learning_rate=0.1, decay_rate=0.001, importance_factor=2.0 , activation_threshold=3000):
        """
        Initialize the NPSOM (Neuroplastic Self-Organizing Map) parameters.

        Parameters:
            grid_size (tuple): Size of the SOM grid (rows, cols)
            input_dim (int): Dimension of input vectors
            learning_rate (float): Initial learning rate for weight updates
            decay_rate (float): Rate at which synaptic weights decay over time
            importance_factor (float): Factor to control the influence of importance map
        """
        np.random.seed(0)
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.sigma = sigma
        self.importance_factor = importance_factor
        self.activation_threshold = activation_threshold

        # Initialize weights randomly in range [0, 1]
        self.weights = np.random.rand(grid_size[0]*grid_size[1], input_dim)
        self.activation_count = np.zeros((grid_size[0]* grid_size[1]))  # Tracks number of activations for each neuron
        self.importance_map = np.zeros((grid_size[0]* grid_size[1]))  # Measures the importance of each neuron
        self.iter_count = np.zeros((grid_size[0]* grid_size[1]))  # Measures the iterations of each neuron
        self.positions = np.array([[x, y] for x in range(grid_size[0]) for y in range(grid_size[1])])
        self.grid_increase = 0

        

    def find_bmu(self, input_vector):
        """
        Find the Best Matching Unit (BMU) for a given input vector using Euclidean distance.

        Parameters:
            input_vector (ndarray): Input vector

        Returns:
            tuple: Indices of the BMU (i, j)

        """
        #spreaded_weights = self.weights.reshape(self.grid_size[0] , self.grid_size[1], input_dim)
        diff = self.weights - input_vector
        dist = np.linalg.norm(diff, axis=1)  # Compute Euclidean distance
        bmu = np.argmin(dist)
        return bmu

    def update_weights(self, bmu, input_vector ,eta , sigma_t):
        """
        Update the weights of the SOM using the winner-takes-all rule.

        Parameters:
            bmu (tuple): Indices of the Best Matching Unit
            input_vector (ndarray): Input vector
        """
        
        self.weights = self.weights.reshape(-1 , self.input_dim)
        for i, position in enumerate(self.positions):
                dist = np.linalg.norm(position - self.positions[bmu])
                neighborhood_function = np.exp(-dist**2 / (2 * sigma_t**2))
                weight_update = eta * neighborhood_function * (input_vector - self.weights[i])
                #print(weight_update.shape)

                self.weights[i] += weight_update

                # Update activation count and importance map
                self.importance_map[i] += neighborhood_function * self.importance_factor
        self.activation_count[bmu] += 1
                # Debugging output to monitor updates
                #if self.activation_count[i, j] % 100 == 0:  # Print every 100 activations
                    #print(f"Neuron ({i},{j}) activated {self.activation_count[i, j]} times.")

    def synaptic_decay(self):
        """
        Apply synaptic decay to the weights. Weights decay over time, making the map more plastic.
        """
        #Min number of iterations per neuron is 5

    

        self.weights -= (self.decay_rate) * self.weights
        


    def memory_forget(self , num = 1 , min_iter = 5):
        num2 = 1
        #delete long term memory
        memoryv = False
        del_indices = None
        if self.weights.shape[0] >  0:
            arr = self.activation_count.flatten() 
            arr2 = self.importance_map.flatten() 
            arr3 = self.iter_count.flatten()
            arr = -1 + 2 * (arr - arr.min()) / (arr.max() - arr.min())
            arr2 = -1 + 2 * (arr2 - arr2.min()) / (arr2.max() - arr2.min())
            #indices = np.where((arr < 1000) & (arr2 < 50))[0]
            #sorted_indices = indices[np.argsort(arr[indices])]
            indices = np.argsort(arr + arr2)[:num*2]
            extra_sorted_indices = indices[(arr3[indices] > min_iter) ]
            del_indices = extra_sorted_indices[:num2] if len(extra_sorted_indices) >= num2 else extra_sorted_indices
        if del_indices is None :
            return memoryv
        else :
            print(f'{len(del_indices)} neuron deleted at {del_indices}')
            self.weights = np.delete(self.weights, del_indices , axis=0).reshape(-1 , self.input_dim)
            self.importance_map = np.delete(self.importance_map , del_indices)
            self.activation_count = np.delete(self.activation_count , del_indices)
            self.iter_count = np.delete(self.iter_count , del_indices)
            self.positions = np.delete(self.positions, del_indices, axis=0)
            self.grid_increase -=num2

        return True



    def insert_neuron(existing_grid, new_neuron, i, j):
        # Split the existing grid at the row `i` and column `j`
        before = existing_grid[:i, :, :]
        after = existing_grid[i:, :, :]
    
        # Insert the new neuron at position (i, j) - between the split parts
        # The new_neuron should be of shape (1, 10, 4)
    
        #    Concatenate before, the new neuron, and after
        new_grid = np.concatenate((before, new_neuron, after), axis=0)
    
        return new_grid

    def neurogenesis(self, batch ,num=1 ):
        """
        Implement neurogenesis. Create new neurons in the SOM grid if necessary.
        Currently, we add neurons with random initialization if the importance map suggests an area with low activity.
        """
        # Add new neurons if certain criteria are met (e.g., low activity in certain regions)
        #flattened_grid = np.prod(self.grid_size)
        #condition for neurogenesis:
        #assumption: 1."""Perform neurogenesis based on combined importance and error."""
        # Compute the score map
        epsilon = 1e-5
        n_clusters = num+1
        #score_map = 1 / (self.importance_map + epsilon)
        # Find the neuron with the maximum score
        #max_score_pos = np.argmax(score_map)
        #i = int(max_score_pos / self.grid_size[1] )
        #j = int(max_score_pos % self.grid_size[1] )
        #pos_tuple = (i, j)
        #print(self.weights[max_score_pos])
        # Choose a neighbor to place the new neuron
        #neighbors = self.get_neighbors(pos_tuple)
        #print(neighbors)
        #new_neuron_pos = self.refine_position(pos_tuple, neighbors)
        #print(new_neuron_pos)
        # Perform K-Means clustering on the batch data
        if n_clusters > len(batch):
            return
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(batch)
        # Get the centroids (cluster centers)
        centroids = kmeans.cluster_centers_
        # Add each centroid as a new neuron in the SOM grid
        sum = 0
        centroids2= []
        sums=[]
        for centroid in centroids :
            sum2 = np.sum(np.linalg.norm(self.weights - centroid, axis=-1) )
            if sum2 > sum : 
                centroids2.append(centroid)
        centroids2 = centroids2
        for centroid in centroids2[:num] :
            new_weight= centroid + 0.1 * (np.random.rand(1 , self.input_dim) )
            self.weights = np.append(self.weights,new_weight)
            self.activation_count = np.append(self.activation_count , 0)
            self.importance_map = np.append(self.importance_map , 0)
            self.iter_count = np.append(self.iter_count , 0)
            self.positions = np.vstack([self.positions, np.random.randint(0, self.grid_size[0], size=(1, 2))])
            print("neuron added")
        self.grid_increase +=num
        self.weights = self.weights.reshape(-1 , self.input_dim)
        '''
        if self.activation_count[i, j] < self.activation_threshold:
            # Create a new neuron
            new_weight = np.random.rand(1 , self.input_dim)  # Initialize new neuron weights randomly
            new_position = (i, j)  # Add it at the current neuron position or near it
                    

            # Update grid to include the new neuron (resize weight matrix)
            self.weights = np.insert(self.weights,i * self.grid_size[1] + j , new_weight, axis=0)
            self.grid_increase +=1
            self.grid_size = (self.grid_size[0] + self.grid_increase%2 , self.grid_size[1] + (self.grid_increase-1)%2)  # Increase grid size

            # Reset activation counts for the new neuron
            self.activation_count = np.insert(self.activation_count,i*self.grid_size[1] + j ,  np.zeros((1,)), axis=None)

            # Optional: print to debug the new neuron creation
            print(f"Created new neuron at position {new_position} with weights {new_weight}")
        '''

    
    def get_neighbors(self, pos):
        """Get neighboring positions for a given neuron."""
        neighbors = []
        for i in range(max(0, pos[0] - 1), min(self.grid_size[0], pos[0] + 2)):
            for j in range(max(0, pos[1] - 1), min(self.grid_size[1], pos[1] + 2)):
                if (i, j) != pos:
                    neighbors.append((i, j))
        return neighbors

    def refine_position(self, pos, neighbors):
        """Refine position of the new neuron based on importance and activity."""
        # Choose the neighbor with the lowest importance
        min_importance_neighbor = min(neighbors, key=lambda x: self.importance_map[x[0]*self.grid_size[1] + self.grid_size[0]])
        return min_importance_neighbor  
    

    def train_batch(self, batch,num , eta , sigma_t , iterations=30 ):
        """Train the SOM for a given batch with multiple iterations."""
        memoryv = False
        for iteration in range(iterations):
            for input_vector in batch:
                bmu = self.find_bmu(input_vector)
                self.update_weights(bmu, input_vector , eta , sigma_t)
                
            
            # Visualize weights after each iteration
        #self.visualize_weights()
        self.synaptic_decay()

    def train(self, data,num , batch_size=10, iterations_per_batch=30 ):
        """Train the SOM with sequential batch updates."""
        # Split the data into batches
        #batches = create_random_batches(data, batch_size, seed=42)
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        memoryv = False
        total_iterations=iterations_per_batch*len(batches)
        # Process each batch with multiple iterations
        for batch_idx, batch in enumerate(batches):
            print(f"Processing Batch {batch_idx + 1}/{len(batches)}")
            eta = self.learning_rate * np.exp(-batch_idx* iterations_per_batch / total_iterations)
            sigma_t = self.sigma * np.exp(-batch_idx* iterations_per_batch  / total_iterations)
            self.train_batch(batch,num , eta , sigma_t, iterations=iterations_per_batch)
            self.iter_count += iterations_per_batch
            memoryv  = True
            if batch_idx % 10 == 0:
                self.memory_forget(num , min_iter=iterations_per_batch*2  )
            #self.visualize_weights()
            
    def calculate_quantization_error(self, data):
        """Calculate the quantization error for the dataset."""
        total_error = 0
        error_arr = []
        for weight in self.weights:
            error = np.sum(np.linalg.norm(data - weight , axis = 1))
            error_arr.append(error)
        return np.array(error_arr)
    
    def visualize_weights(self):
        """Improved visualization of SOM weights using a heatmap."""

        rows, cols = find_best_grid_shape(self.weights.shape[0])


        # Ensure correct reshaping
        importance_map2 = np.append(self.importance_map , np.zeros(rows*cols - self.weights.shape[0]))
        importance_map = importance_map2.reshape(rows, cols, -1)


        # Create X, Y meshgrid
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

        # Compute importance map (Z-axis) using L2 norm
        Z = np.linalg.norm(importance_map, axis=2)


        Z_smooth = zoom(Z, 3)  # Increase resolution by 3x

        # Create high-resolution X, Y meshgrid
        X_smooth = np.linspace(0, cols-1, Z_smooth.shape[1])
        Y_smooth = np.linspace(0, rows-1, Z_smooth.shape[0])
        X_smooth, Y_smooth = np.meshgrid(X_smooth, Y_smooth)

        # Plot in 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X_smooth, Y_smooth, Z_smooth, cmap="viridis", edgecolor='none',  antialiased=True)

        fig.colorbar(surf, shrink=0.5, aspect=10, label="Weight Importance (L2 Norm)")
        ax.set_xlabel("Grid X (Columns)")
        ax.set_ylabel("Grid Y (Rows)")
        ax.set_zlabel("Importance (Weight Norm)")
        ax.set_title("3D SOM Weight Visualization")

        plt.show()
  
    def assign_labels_to_neurons(self, data, labels , k = 2):
        """Assign labels to neurons based on majority voting."""
        neuron_labels = {}
        for i, weight in enumerate(self.weights):
            distances = np.linalg.norm(data - weight, axis=1)
            nearest_indices = np.argsort(distances)[:k+1]
            nearest_labels = labels[nearest_indices]
        
            # Assign the majority label to the neuron
            if len(nearest_labels) > 0:
                neuron_labels[i] = Counter(nearest_labels).most_common(1)[0][0]
            else:
                neuron_labels[i] = None 
        return neuron_labels


    def predict(self, data, neuron_labels):
        """Predict the class of each data point."""
        predictions = []
        for input_vector in data:
            bmu_index = self.find_bmu(input_vector)
            predictions.append(neuron_labels.get(bmu_index, -1) )  # -1 if BMU has no label
        return predictions
    
    def calculate_importance_map(self, data):
        """
        Calculate importance map based on active neurons and their densities.
        Returns a dictionary with neuron indices as keys and normalized densities as values.
        """
        # Track the density of each active neuron
        neuron_density = defaultdict(int)

        # Map data points to the active neurons
        for input_vector in data:
            bmu_index = self.find_bmu(input_vector)  # Use index for dynamic neurons
            neuron_density[bmu_index] += 1

        # Normalize densities
        max_density = max(neuron_density.values(), default=1)
        importance_map = {neuron: density / max_density for neuron, density in neuron_density.items()}

        return importance_map


def rank_features_by_variance(som):
    """Rank features by their importance using variance across neuron weights."""
    # Calculate variance of each feature across all neurons
    feature_variances = np.var(som.weights, axis=0)  # Variance along the feature dimension
    # Rank features in descending order of variance
    feature_ranking = np.argsort(-feature_variances)  # Indices of features ranked by importance
    return feature_ranking, feature_variances


def rank_features_by_activation(som):
    activation_count = som.activation_count
    top_neurons  = np.argsort(activation_count)
    feature_contributions = np.var(som.weights[top_neurons, :], axis=0)
    feature_ranking = np.argsort(-feature_contributions)  # Descending order of contributions

    return feature_ranking, feature_contributions

def rank_features_by_activation2(som):
    activation_count = som.activation_count/som.iter_count
    top_neurons  = np.argsort(-activation_count)[:30]
    feature_contributions = np.var(som.weights[top_neurons, :], axis=0)
    feature_ranking = np.argsort(-feature_contributions)  # Descending order of contributions

    return feature_ranking, feature_contributions

def rank_features_by_importance(som):
    activation_count = som.importance_map /som.iter_count
    num_neurons = activation_count.shape[0]
    top_neurons  = np.argsort(-activation_count)[:num_neurons*3//4]
    feature_contributions = np.var(som.weights[top_neurons, :], axis=0)
    feature_ranking = np.argsort(-feature_contributions)  # Descending order of contributions

    return feature_ranking, feature_contributions

def rank_features_by_hybridapproach(som , alpha , beta , gamma):
    scaler = MinMaxScaler()

    # Fit and transform the array
    importance_map = scaler.fit_transform(som.importance_map.reshape(-1, 1)).flatten()
    activation_count = scaler.fit_transform(som.activation_count.reshape(-1, 1)).flatten()
    activation_count = 100*(alpha * importance_map + beta*activation_count )/gamma* som.iter_count
    num_neurons = activation_count.shape[0]
    top_neurons  = np.argsort(-activation_count)[:num_neurons*3//4]
    feature_contributions = np.var(som.weights[top_neurons, :], axis=0)
    feature_ranking = np.argsort(-feature_contributions)  # Descending order of contributions

    return feature_ranking, feature_contributions



