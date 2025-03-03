import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


"-------------POISSON SURFACE RECON-------------"
def construct_laplacian(points):
    """
    Construct the Laplacian matrix from the input point cloud.
    """
    # Construct a KDTree for efficient nearest neighbor search
    from sklearn.neighbors import NearestNeighbors
    kdtree = NearestNeighbors(n_neighbors=10).fit(points)
    
    # Query the k-nearest neighbors for each point
    k = 10  # Number of neighbors to consider
    distances, indices = kdtree.kneighbors(points, n_neighbors=k+1)
    
    # Initialize a sparse matrix to store the Laplacian coefficients
    n = len(points)
    L = csr_matrix((n, n), dtype=np.float)
    
    # Compute Laplacian coefficients
    for i in range(n):
        for j in indices[i][1:]:
            d = np.linalg.norm(points[i] - points[j])
            L[i, j] = -1 / d
            L[i, i] += 1 / d
    
    return L

def reconstruct_surface(points):
    """
    Reconstruct the surface from the input point cloud using Poisson Surface Reconstruction (PSR).
    """
    # Construct the Laplacian matrix
    L = construct_laplacian(points)
    
    # Solve the Poisson equation: L * phi = div(V)
    V = np.ones(len(points))  # Assume uniform vector field for simplicity
    phi = spsolve(L, np.gradient(V))
    
    # Extract surface points from zero-crossings of phi
    surface_points = [points[i] for i in range(len(points)) if phi[i] == 0]
    
    return surface_points

# Example usage:
# Replace 'points' with your actual point cloud data
points = np.random.rand(100, 3)  # Random example points
surface_points = reconstruct_surface(points)
print("Surface points:", surface_points)


"---------------OCTREE-----------------"
class OctreeNode:
    def __init__(self, center, size):
        self.center = center  # Center of the octant
        self.size = size      # Size of the octant
        self.children = [None] * 8  # Initialize with 8 children
        self.points = []

class Octree:
    def __init__(self, min_size):
        self.root = None
        self.min_size = min_size

    def insert(self, point):
        if self.root is None:
            # Create root node centered at origin with initial size
            self.root = OctreeNode(center=(0, 0, 0), size=float('inf'))
        
        self._insert_recursive(self.root, point, self.min_size)

    def _insert_recursive(self, node, point, min_size):
        if node.size <= min_size:
            # Reached minimum size, add point to this node
            node.points.append(point)
            return
        
        # Determine which octant the point belongs to
        octant_index = self._get_octant_index(node.center, point)
        
        # If the child octant doesn't exist, create it
        if node.children[octant_index] is None:
            # Calculate size and center of the child octant
            child_size = node.size / 2
            child_center = [
                node.center[0] + (child_size / 2) * ((octant_index & 1) * 2 - 1),
                node.center[1] + (child_size / 2) * ((octant_index & 2) - 1),
                node.center[2] + (child_size / 2) * ((octant_index & 4) / 2 - 1)
            ]
            node.children[octant_index] = OctreeNode(center=child_center, size=child_size)
        
        # Recur into the appropriate child octant
        self._insert_recursive(node.children[octant_index], point, min_size)

    def _get_octant_index(self, center, point):
        index = 0
        if point[0] > center[0]:
            index |= 1
        if point[1] > center[1]:
            index |= 2
        if point[2] > center[2]:
            index |= 4
        return index

    def query(self, query_point, radius):
        if self.root is None:
            return []
        return self._query_recursive(self.root, query_point, radius)

    def _query_recursive(self, node, query_point, radius):
        result = []
        if node.size <= radius:
            # Add all points in this node
            result.extend(node.points)
        else:
            # Check children
            for child in node.children:
                if child is not None:
                    if self._intersects(child.center, child.size, query_point, radius):
                        result.extend(self._query_recursive(child, query_point, radius))
        return result

    def _intersects(self, center, size, query_point, radius):
        dx = abs(query_point[0] - center[0])
        dy = abs(query_point[1] - center[1])
        dz = abs(query_point[2] - center[2])
        d = max(dx - size / 2, 0) ** 2 + max(dy - size / 2, 0) ** 2 + max(dz - size / 2, 0) ** 2
        return d <= radius ** 2

# Example usage:
octree = Octree(min_size=1.0)
points = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]  # Example points
for point in points:
    octree.insert(point)

# Query points within a radius
query_point = (2, 3, 4)
radius = 2.0
result = octree.query(query_point, radius)
print("Points within radius:", result)
