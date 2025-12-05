import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionVisualizer:
    """
    Visualizer untuk training process Logistic Regression
    Membuat video loss contour dan parameter trajectory
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialize visualizer

        Args:
            model: Trained LogisticRegression model
            X_train: Training features
            y_train: Training labels
        """
        self.model = model
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.loss_history = model.get_loss_history()

    def plot_loss_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot training loss curve

        Args:
            save_path: Path untuk menyimpan plot (optional)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
        plt.title('Training Loss over Iterations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curve saved to {save_path}")

        plt.show()

    def plot_accuracy_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot training accuracy curve

        Args:
            save_path: Path untuk menyimpan plot (optional)
        """
        accuracy_history = self.model.get_accuracy_history()
        plt.figure(figsize=(10, 6))
        plt.plot(accuracy_history, linewidth=2, color='green')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training Accuracy over Iterations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy curve saved to {save_path}")
        plt.show()

    def _compute_loss_for_params(self, theta0: float, theta1: float,
                                 feature_idx: int = 0) -> float:
        """
        Compute loss untuk specific parameter values

        Args:
            theta0: Bias value
            theta1: Weight value untuk feature tertentu
            feature_idx: Index feature yang akan divariasikan

        Returns:
            Loss value
        """
        X_single = self.X_train[:, feature_idx:feature_idx+1]
        linear_output = X_single * theta1 + theta0
        linear_output = np.clip(linear_output, -500, 500)
        y_pred = 1 / (1 + np.exp(-linear_output))
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Compute loss
        loss = -np.mean(self.y_train * np.log(y_pred) +
                       (1 - self.y_train) * np.log(1 - y_pred))
        return loss

    def create_loss_contour_video(self, save_path: str = 'loss_contour_animation.mp4',
                                 feature_idx: int = 0,
                                 fps: int = 10,
                                 grid_points: int = 50) -> None:
        """
        Create video showing loss contour dan parameter trajectory
        BONUS: Video garis kontur fungsi loss dan lintasan parameter (θ₀, θ₁)

        Args:
            save_path: Path untuk menyimpan video
            feature_idx: Index feature untuk visualisasi (karena 2D)
            fps: Frames per second
            grid_points: Number of grid points untuk contour
        """
        print("Creating loss contour video...")
        print("This may take a few minutes...")

        # Ambil weight trajectory dari history
        # Karena model tidak menyimpan weight history, harus retrain
        # atau menggunakan approximation
        print("\nRetraining model to capture weight trajectory...")

        # Retrain dengan verbose untuk capture trajectory
        weights_history = []
        bias_history = []
        from .logres import LogisticRegression
        temp_model = LogisticRegression(
            learning_rate=self.model.learning_rate,
            n_iterations=self.model.n_iterations,
            regularization=self.model.regularization,
            lambda_reg=self.model.lambda_reg,
            verbose=False
        )

        # Manual training loop dengan weight tracking
        X = self.X_train
        y = self.y_train
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0
        weights_history.append(weights.copy())
        bias_history.append(bias)

        for iteration in range(temp_model.n_iterations):
            # Forward pass
            linear_output = np.dot(X, weights) + bias
            linear_output = np.clip(linear_output, -500, 500)
            y_pred = 1 / (1 + np.exp(-linear_output))

            # Compute gradients
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            weights -= temp_model.learning_rate * dw
            bias -= temp_model.learning_rate * db

            # Save every few iterations untuk animation
            if iteration % max(1, temp_model.n_iterations // 100) == 0:
                weights_history.append(weights.copy())
                bias_history.append(bias)

        weights_history = np.array(weights_history)
        bias_history = np.array(bias_history)
        print(f"Captured {len(weights_history)} parameter snapshots")

        # Create grid untuk loss contour
        theta1_history = weights_history[:, feature_idx]

        # Define parameter range
        theta0_min, theta0_max = bias_history.min() - 0.5, bias_history.max() + 0.5
        theta1_min, theta1_max = theta1_history.min() - 0.5, theta1_history.max() + 0.5
        theta0_grid = np.linspace(theta0_min, theta0_max, grid_points)
        theta1_grid = np.linspace(theta1_min, theta1_max, grid_points)

        # Compute loss contour
        print("Computing loss contour...")
        loss_grid = np.zeros((grid_points, grid_points))

        for i, theta0 in enumerate(theta0_grid):
            for j, theta1 in enumerate(theta1_grid):
                loss_grid[i, j] = self._compute_loss_for_params(theta0, theta1, feature_idx)

        # Create animation
        print("Creating animation...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot contour
        contour = ax.contour(theta1_grid, theta0_grid, loss_grid, levels=20, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)

        # Setup plot
        ax.set_xlabel(f'θ₁ (Weight for feature {feature_idx})', fontsize=12)
        ax.set_ylabel('θ₀ (Bias)', fontsize=12)
        ax.set_title('Loss Contour and Parameter Trajectory', fontsize=14, fontweight='bold')

        # Initialize line dan point untuk trajectory
        line, = ax.plot([], [], 'r-', linewidth=2, label='Parameter Path')
        point, = ax.plot([], [], 'ro', markersize=10, label='Current Position')
        ax.legend()

        # Animation function
        def animate(frame):
            # Update trajectory
            line.set_data(theta1_history[:frame+1], bias_history[:frame+1])
            point.set_data([theta1_history[frame]], [bias_history[frame]])
            ax.set_title(f'Loss Contour and Parameter Trajectory (Iteration {frame * max(1, temp_model.n_iterations // 100)})',
                        fontsize=14, fontweight='bold')
            return line, point

        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(weights_history),
                           interval=1000/fps, blit=True, repeat=True)

        print(f"Saving video to {save_path}...")
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Video saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving video with FFMpegWriter: {e}")
            print("Trying alternative method...")
            try:
                anim.save(save_path, fps=fps, bitrate=1800)
                print(f"Video saved successfully to {save_path}")
            except Exception as e2:
                print(f"Error: {e2}")
                print("Please install ffmpeg: conda install -c conda-forge ffmpeg")
        plt.close()

    def create_simple_loss_surface_plot(self, save_path: Optional[str] = None,
                                       feature_idx: int = 0,
                                       grid_points: int = 50) -> None:
        """
        Create static 3D plot of loss surface dengan parameter trajectory

        Args:
            save_path: Path untuk menyimpan plot
            feature_idx: Index feature untuk visualisasi
            grid_points: Number of grid points
        """
        from mpl_toolkits.mplot3d import Axes3D

        print("Creating 3D loss surface plot...")

        # Get final parameters
        final_bias = self.model.bias
        final_weight = self.model.weights[feature_idx]

        # Define parameter range
        theta0_range = np.linspace(final_bias - 2, final_bias + 2, grid_points)
        theta1_range = np.linspace(final_weight - 2, final_weight + 2, grid_points)

        theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)

        # Compute loss surface
        loss_surface = np.zeros_like(theta0_grid)
        for i in range(grid_points):
            for j in range(grid_points):
                loss_surface[i, j] = self._compute_loss_for_params(
                    theta0_grid[i, j], theta1_grid[i, j], feature_idx
                )

        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(theta1_grid, theta0_grid, loss_surface,
                              cmap='viridis', alpha=0.7, edgecolor='none')

        # Plot final parameter point
        final_loss = self._compute_loss_for_params(final_bias, final_weight, feature_idx)
        ax.scatter([final_weight], [final_bias], [final_loss],
                  color='red', s=100, marker='o', label='Final Parameters')

        # Labels
        ax.set_xlabel(f'θ₁ (Weight for feature {feature_idx})', fontsize=11)
        ax.set_ylabel('θ₀ (Bias)', fontsize=11)
        ax.set_zlabel('Loss', fontsize=11)
        ax.set_title('Loss Surface', fontsize=14, fontweight='bold')

        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D loss surface saved to {save_path}")
        plt.show()

    def plot_decision_boundary(self, feature_idx1: int = 0, feature_idx2: int = 1,
                              save_path: Optional[str] = None) -> None:
        """
        Plot decision boundary (hanya untuk 2 features)

        Args:
            feature_idx1: Index feature 1
            feature_idx2: Index feature 2
            save_path: Path untuk menyimpan plot
        """
        if self.X_train.shape[1] < 2:
            print("Need at least 2 features to plot decision boundary")
            return

        # Ambil 2 features
        X_2d = self.X_train[:, [feature_idx1, feature_idx2]]

        # Create meshgrid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))

        # Predict untuk setiap point di mesh
        # Buat dummy features untuk features lain (set ke mean)
        mesh_samples = np.c_[xx.ravel(), yy.ravel()]
        if self.X_train.shape[1] > 2:
            other_features = np.tile(self.X_train.mean(axis=0), (mesh_samples.shape[0], 1))
            other_features[:, [feature_idx1, feature_idx2]] = mesh_samples
            mesh_samples = other_features
        Z = self.model.predict(mesh_samples)
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.y_train,
                   edgecolors='k', cmap='RdYlBu', s=50)

        plt.xlabel(f'Feature {feature_idx1}', fontsize=12)
        plt.ylabel(f'Feature {feature_idx2}', fontsize=12)
        plt.title('Decision Boundary', fontsize=14, fontweight='bold')
        plt.colorbar(label='Predicted Class')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decision boundary saved to {save_path}")
        plt.show()