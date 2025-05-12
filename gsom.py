import numpy as np
from sklearn.preprocessing import StandardScaler

class GSOM:
    def __init__(self, input_dim, spread_factor=0.9, learning_rate=0.1, max_iter=1000):
        self.input_dim = input_dim
        self.spread_factor = spread_factor
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.grid = {}  # Stores nodes (weights of GSOM)

    def initialize_grid(self):
        """Initialize a 2x2 starting grid with random weights"""
        self.grid = {
            (0,0): np.random.rand(self.input_dim),
            (0,1): np.random.rand(self.input_dim),
            (1,0): np.random.rand(self.input_dim),
            (1,1): np.random.rand(self.input_dim),
        }
    
    def find_bmu(self, x):
        """Find Best Matching Unit (BMU)"""
        return min(self.grid, key=lambda k: np.linalg.norm(self.grid[k] - x))
    
    def grow(self, bmu):
        """Grow new nodes around the BMU if required"""
        neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
        for dx, dy in neighbors:
            new_pos = (bmu[0] + dx, bmu[1] + dy)
            if new_pos not in self.grid:
                self.grid[new_pos] = self.grid[bmu] + np.random.rand(self.input_dim) * 0.1  # Small variation
    
    def train(self, data):
        """Train the GSOM model"""
        self.initialize_grid()
        for i in range(self.max_iter):
            x = data[np.random.randint(0, len(data))]
            bmu = self.find_bmu(x)
            self.grid[bmu] += self.learning_rate * (x - self.grid[bmu])  # Update BMU weight
            self.grow(bmu)  # Grow if necessary

    def get_feature_reduction(self , numoffeatures = 300):
        """Select important features based on GSOM weights"""
        # Aggregate weights across all nodes
        weight_matrix = np.array(list(self.grid.values()))  # Convert GSOM nodes to matrix
        
        # Compute feature importance based on weight variance
        feature_importance = np.var(weight_matrix, axis=0)
        
        # Select top-k features with highest variance (k = input_dim // 2)
        selected_features = np.argsort(feature_importance)[-100:]
        
        return selected_features

