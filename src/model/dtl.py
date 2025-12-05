import numpy as np
from typing import Dict, Any, Union, List, Optional
from collections import Counter
from .base_model import BaseModel

class Node:
    """node untuk decision tree"""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree(BaseModel):
    """implementasi decision tree (cart) from scratch"""
    def __init__(self, min_samples_split=2, max_depth=100, criterion='gini'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None
        self.imputation_values = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """melatih decision tree dengan data training"""
        X = np.array(X, dtype=object)
        y = np.array(y)
        X = self._handle_missing_values_fit(X)
        dataset = np.column_stack((X, y))
        self.root = self._grow_tree(dataset)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """prediksi label untuk data baru"""
        X = np.array(X, dtype=object)
        X = self._handle_missing_values_predict(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def get_params(self) -> Dict[str, Any]:
        """mengambil parameter model"""
        return {
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "imputation_values": self.imputation_values
        }

    def _handle_missing_values_fit(self, X: np.ndarray) -> np.ndarray:
        """isi missing values dengan modus saat training"""
        n_features = X.shape[1]
        for i in range(n_features):
            col = X[:, i]
            valid_mask = [pd_isna(val) == False for val in col]
            valid_values = col[valid_mask]

            if len(valid_values) > 0:
                mode_val = Counter(valid_values).most_common(1)[0][0]
                self.imputation_values[i] = mode_val

                for idx in range(len(col)):
                    if pd_isna(col[idx]):
                        X[idx, i] = mode_val
            else:
                self.imputation_values[i] = 0

        return X

    def _handle_missing_values_predict(self, X: np.ndarray) -> np.ndarray:
        """isi missing values saat prediksi dengan nilai dari training"""
        n_features = X.shape[1]
        for i in range(n_features):
            if i in self.imputation_values:
                fill_val = self.imputation_values[i]
                for idx in range(len(X)):
                    if pd_isna(X[idx, i]):
                        X[idx, i] = fill_val
        return X

    def _grow_tree(self, dataset: np.ndarray, depth=0) -> Node:
        """membangun tree secara rekursif"""
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # optimasi: early stopping untuk node yang hampir pure (>= 95%)
        most_common_count = max([np.sum(y == cls) for cls in np.unique(y)])
        purity = most_common_count / len(y)
        if purity >= 0.95:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        best_split = self._get_best_split(dataset, n_samples, n_features)

        if best_split.get("gain", 0) > 0:
            left_subtree = self._grow_tree(best_split["dataset_left"], depth + 1)
            right_subtree = self._grow_tree(best_split["dataset_right"], depth + 1)
            return Node(feature_index=best_split["feature_index"],
                        threshold=best_split["threshold"],
                        left=left_subtree,
                        right=right_subtree)

        return Node(value=self._calculate_leaf_value(y))

    def _get_best_split(self, dataset: np.ndarray, n_samples: int, n_features: int) -> Dict:
        """cari split terbaik untuk semua fitur"""
        best_split = {}
        max_gain = -float("inf")
        y = dataset[:, -1]

        for feature_index in range(n_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # optimasi: batasi kandidat threshold menggunakan percentiles
            if len(possible_thresholds) > 50 and not isinstance(possible_thresholds[0], str):
                possible_thresholds = np.percentile(feature_values, np.linspace(0, 100, 50))
                possible_thresholds = np.unique(possible_thresholds)

            for threshold in possible_thresholds:
                # optimasi: hitung gain tanpa copy array dulu (lazy copying)
                if isinstance(threshold, str):
                    left_mask = feature_values == threshold
                else:
                    left_mask = feature_values <= threshold

                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]
                curr_gain = self._information_gain(y, left_y, right_y)

                if curr_gain > max_gain:
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                        "gain": curr_gain
                    }
                    max_gain = curr_gain

        # copy array hanya untuk best split (efisiensi memori)
        if best_split:
            left_mask = best_split.pop("left_mask")
            right_mask = best_split.pop("right_mask")
            best_split["dataset_left"] = dataset[left_mask]
            best_split["dataset_right"] = dataset[right_mask]

        return best_split

    def _split(self, dataset: np.ndarray, feature_index: int, threshold: Any):
        """pisahkan dataset berdasarkan threshold (numerik/kategorikal)"""
        feature_values = dataset[:, feature_index]

        if isinstance(threshold, str):
            left_indices = np.where(feature_values == threshold)[0]
            right_indices = np.where(feature_values != threshold)[0]
        else:
            left_indices = np.where(feature_values <= threshold)[0]
            right_indices = np.where(feature_values > threshold)[0]

        return dataset[left_indices], dataset[right_indices]

    def _information_gain(self, parent, l_child, r_child):
        """hitung information gain dari split"""
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if self.criterion == "gini":
            return self._gini(parent) - (weight_l * self._gini(l_child) + weight_r * self._gini(r_child))
        else:
            return self._entropy(parent) - (weight_l * self._entropy(l_child) + weight_r * self._entropy(r_child))

    def _entropy(self, y):
        """hitung entropy untuk split"""
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _gini(self, y):
        """hitung gini impurity untuk split"""
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _calculate_leaf_value(self, y):
        """hitung nilai prediksi untuk leaf node (majority voting)"""
        y = list(y)
        return max(y, key=y.count)

    def _traverse_tree(self, x: np.ndarray, node: Node):
        """traverse tree untuk prediksi satu sampel"""
        if node.value is not None:
            return node.value

        feature_val = x[node.feature_index]

        if isinstance(node.threshold, str):
            if feature_val == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if feature_val <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def visualize_tree(self, feature_names: Optional[List[str]] = None,
                       max_depth: Optional[int] = None,
                       filename: str = "decision_tree.png",
                       figsize: tuple = None):
        """
        Visualisasi struktur decision tree

        Parameters:
        -----------
        feature_names : list of str, optional
            Nama-nama fitur untuk labeling node
        max_depth : int, optional
            Maksimum kedalaman tree yang ditampilkan (top-N branches)
            Jika None, tampilkan seluruh tree
        filename : str
            Nama file output (default: decision_tree.png)
        figsize : tuple
            Ukuran figure (width, height) - auto calculated if None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Error: matplotlib tidak tersedia. Install dengan: pip install matplotlib")
            return

        if self.root is None:
            print("Error: Tree belum dilatih. Panggil fit() terlebih dahulu.")
            return

        # Hitung kedalaman tree untuk auto sizing
        tree_depth = self._get_tree_depth(self.root, max_depth)
        tree_width = self._count_leaves(self.root, max_depth)

        # Auto calculate figsize based on tree size
        if figsize is None:
            width = max(20, min(tree_width * 1.5, 100))
            height = max(12, tree_depth * 2.5)
            figsize = (width, height)

        # Hitung posisi semua node dengan algoritma yang lebih baik
        positions = {}
        self._calculate_positions_improved(self.root, 0, 0, tree_width, positions, max_depth=max_depth)

        if not positions:
            print("Error: Tidak ada node untuk divisualisasikan")
            return

        # Buat figure
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate dynamic limits
        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Set limits with adequate padding
        padding_x = max(1.0, (max_x - min_x) * 0.05)
        padding_y = 1.0
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
        ax.axis('off')

        # Gambar edges terlebih dahulu
        self._draw_edges(ax, self.root, positions, max_depth=max_depth)

        # Gambar nodes
        self._draw_nodes(ax, self.root, positions, feature_names, max_depth=max_depth)

        plt.title("Decision Tree Visualization", fontsize=16, fontweight='bold', pad=20)

        # Tambahkan legend
        if max_depth is not None:
            legend_text = f"Showing top {max_depth} levels"
            ax.text(0.5, 0.95, transform=ax.transAxes,
                   s=legend_text, fontsize=10, ha='center', style='italic')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Tree visualization saved to: {filename}")
        plt.close()

    def _get_tree_depth(self, node: Node, max_depth: Optional[int] = None, current_depth: int = 0) -> int:
        """Hitung kedalaman maksimum tree"""
        if node is None or (max_depth is not None and current_depth >= max_depth):
            return current_depth

        if node.value is not None:  # Leaf node
            return current_depth

        left_depth = self._get_tree_depth(node.left, max_depth, current_depth + 1)
        right_depth = self._get_tree_depth(node.right, max_depth, current_depth + 1)
        return max(left_depth, right_depth)

    def _count_leaves(self, node: Node, max_depth: Optional[int] = None, current_depth: int = 0) -> int:
        """Hitung jumlah leaf nodes"""
        if node is None or (max_depth is not None and current_depth > max_depth):
            return 0

        if node.value is not None or (max_depth is not None and current_depth == max_depth):
            return 1

        left_count = self._count_leaves(node.left, max_depth, current_depth + 1)
        right_count = self._count_leaves(node.right, max_depth, current_depth + 1)
        return left_count + right_count

    def _calculate_positions_improved(self, node: Node, depth: int, x: float, width: float,
                                     positions: dict, max_depth: Optional[int] = None):
        """
        Hitung posisi x,y untuk setiap node dengan algoritma yang lebih baik
        Menggunakan spacing yang proporsional dengan jumlah leaves di subtree
        """
        if node is None or (max_depth is not None and depth > max_depth):
            return 0

        # Vertical spacing
        y = -depth * 2.5  # Increased vertical spacing

        if node.value is not None or (max_depth is not None and depth == max_depth):
            # Leaf node
            positions[id(node)] = (x, y, depth)
            return 1

        # Hitung jumlah leaves untuk setiap subtree
        left_leaves = self._count_leaves(node.left, max_depth, depth + 1) if node.left else 0
        right_leaves = self._count_leaves(node.right, max_depth, depth + 1) if node.right else 0
        total_leaves = max(1, left_leaves + right_leaves)

        # Hitung posisi untuk children dengan spacing proporsional
        if node.left is not None and (max_depth is None or depth < max_depth):
            left_width = (width * left_leaves / total_leaves)
            self._calculate_positions_improved(
                node.left, depth + 1, x - left_width/2, left_width,
                positions, max_depth
            )

        if node.right is not None and (max_depth is None or depth < max_depth):
            right_width = (width * right_leaves / total_leaves)
            self._calculate_positions_improved(
                node.right, depth + 1, x + right_width/2, right_width,
                positions, max_depth
            )

        # Posisi node saat ini adalah di tengah antara children
        if node.left and node.right and id(node.left) in positions and id(node.right) in positions:
            left_x = positions[id(node.left)][0]
            right_x = positions[id(node.right)][0]
            x = (left_x + right_x) / 2
        elif node.left and id(node.left) in positions:
            x = positions[id(node.left)][0]
        elif node.right and id(node.right) in positions:
            x = positions[id(node.right)][0]

        positions[id(node)] = (x, y, depth)
        return total_leaves

    def _calculate_positions(self, node: Node, depth: int, x: float, width: float,
                            positions: dict, max_depth: Optional[int] = None):
        """Hitung posisi x,y untuk setiap node"""
        if node is None or (max_depth is not None and depth > max_depth):
            return

        y = -depth
        positions[id(node)] = (x, y, depth)

        if node.left is not None and (max_depth is None or depth < max_depth):
            self._calculate_positions(node.left, depth + 1, x - width/2, width/2,
                                     positions, max_depth)

        if node.right is not None and (max_depth is None or depth < max_depth):
            self._calculate_positions(node.right, depth + 1, x + width/2, width/2,
                                      positions, max_depth)

    def _draw_edges(self, ax, node: Node, positions: dict, max_depth: Optional[int] = None):
        """Gambar edges (garis penghubung antar node)"""
        if node is None or id(node) not in positions:
            return

        x, y, depth = positions[id(node)]

        if max_depth is not None and depth >= max_depth:
            return

        # Draw edge ke left child
        if node.left is not None and id(node.left) in positions:
            x_left, y_left, _ = positions[id(node.left)]
            ax.plot([x, x_left], [y, y_left], 'k-', linewidth=1.5, alpha=0.6)
            # Label untuk edge
            mid_x, mid_y = (x + x_left) / 2, (y + y_left) / 2
            ax.text(mid_x - 0.02, mid_y, 'True', fontsize=8, ha='right',
                   style='italic', color='green')
            self._draw_edges(ax, node.left, positions, max_depth)

        # Draw edge ke right child
        if node.right is not None and id(node.right) in positions:
            x_right, y_right, _ = positions[id(node.right)]
            ax.plot([x, x_right], [y, y_right], 'k-', linewidth=1.5, alpha=0.6)
            # Label untuk edge
            mid_x, mid_y = (x + x_right) / 2, (y + y_right) / 2
            ax.text(mid_x + 0.02, mid_y, 'False', fontsize=8, ha='left',
                   style='italic', color='red')
            self._draw_edges(ax, node.right, positions, max_depth)

    def _draw_nodes(self, ax, node: Node, positions: dict,
                   feature_names: Optional[List[str]] = None,
                   max_depth: Optional[int] = None):
        """Gambar nodes (kotak berisi informasi split/leaf)"""
        if node is None or id(node) not in positions:
            return

        x, y, depth = positions[id(node)]

        # Tentukan warna dan ukuran node
        if node.value is not None:  # Leaf node
            box_color = '#90EE90'  # Light green
            box_width, box_height = 0.20, 0.12
        else:  # Internal node
            box_color = '#87CEEB'  # Sky blue
            box_width, box_height = 0.30, 0.15

        # Gambar rectangle untuk node
        import matplotlib.patches as patches
        rect = patches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.005",
            linewidth=1.5,
            edgecolor='black',
            facecolor=box_color,
            alpha=0.8
        )
        ax.add_patch(rect)

        # Text untuk node
        if node.value is not None:  # Leaf node
            text = f"Class: {node.value}"
            ax.text(x, y, text, ha='center', va='center',
                   fontsize=10, fontweight='bold', wrap=True)
        else:  # Internal node
            if feature_names and node.feature_index < len(feature_names):
                feature_name = feature_names[node.feature_index]
            else:
                feature_name = f"X[{node.feature_index}]"

            if isinstance(node.threshold, str):
                text = f"{feature_name}\n== '{node.threshold}'"
            else:
                text = f"{feature_name}\n<= {node.threshold:.3f}"

            ax.text(x, y, text, ha='center', va='center',
                   fontsize=9, multialignment='center', wrap=True)

        # Rekursif untuk child nodes
        if max_depth is None or depth < max_depth:
            if node.left is not None:
                self._draw_nodes(ax, node.left, positions, feature_names, max_depth)
            if node.right is not None:
                self._draw_nodes(ax, node.right, positions, feature_names, max_depth)

def pd_isna(obj):
    """cek apakah nilai adalah nan/none"""
    return obj != obj or obj is None

def regenerate_tree_visualization(model_path: str, output_file: str = 'src/dtl.png',
                                   max_depth: int = 10) -> None:
    """
    Utility function untuk regenerate visualisasi decision tree dari model yang sudah disimpan

    Parameters:
    -----------
    model_path : str
        Path ke file model (.pkl)
    output_file : str
        Path untuk output file visualisasi (default: 'src/dtl.png')
    max_depth : int
        Maksimum kedalaman tree yang ditampilkan (default: 10)
        Set None untuk menampilkan seluruh tree
    """
    from ..storage.model_storage import ModelStorage

    print(f"Loading model from {model_path}...")
    model = ModelStorage.load_model_for_prediction(model_path)

    print("Model loaded successfully!")
    print(f"Model type: {model.__class__.__name__}")

    print(f"\nGenerating improved visualization to {output_file}...")
    print("This may take a moment for large trees...")

    # Visualize with limited depth to avoid extreme size
    model.visualize_tree(
        max_depth=max_depth,
        filename=output_file
    )

    print("\nDone! The visualization has been regenerated with improved spacing.")
    print(f"Check the file: {output_file}")