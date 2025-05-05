import numpy as np
from gen_run import *
from typing import Tuple
import scipy.ndimage
from bez_traj import *


num_steps = 30
class MPPI():
    def __init__(self,u_prev,horizon_step_T,param_exploration=0.01):
        self.u_prev = u_prev
        self.flag_done = False
        self.K = 1
        self.T = horizon_step_T
        sigma_value = 0.9  # Variance of each control input (adjustable)
        self.dim_u = 6
        self.sigma_y = np.diag([0.017, 0.017])  
        self.sigma_z = np.diag([0.017, 0.017]) 
        self.mu_y = np.array([-0.3, -0.3])
        self.mu_z = np.array([0.3, 0.3])
        #self.Sigma = np.diag([sigma_value] *self.dim_u )
        self.param_exploration = param_exploration
        self.param_lambda = 50.0
        self.prev_waypoints_idx = 0
        self.count = 0

    def get_nearest_reference_points(self, x_t,update_prev_idx: bool = False):
        """
        Finds the nearest reference (x, y, z) in trajectory_bezier for each of the two corners.

        Parameters:
        - trajectory_bezier: np.ndarray of shape (2, 50, 3), representing two corners with 50 waypoints.
        - x, y: float, representing the current position.

        Returns:
        - ref_points: List of two tuples (ref_x, ref_y, ref_z) for each corner.
        """
        SEARCH_IDX_LEN = 10 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        ref_points = []  # Store (ref_x, ref_y, ref_z) for both corners
        x = x_t[:,0]
        y = x_t[:,1]
        z = x_t[:,2]
        for corner_idx in range(2):  # Iterate over both corners
            dx =[x[corner_idx]-ref_x for ref_x in trajectory_bezier[corner_idx,prev_idx:(prev_idx+SEARCH_IDX_LEN),0]]
            dy =[y[corner_idx]-ref_y for ref_y in trajectory_bezier[corner_idx,prev_idx:(prev_idx+SEARCH_IDX_LEN),1]]
            dz =[z[corner_idx]-ref_z for ref_z in trajectory_bezier[corner_idx,prev_idx:(prev_idx+SEARCH_IDX_LEN),2]]
            distances = [idx ** 2 + idy ** 2+ idz**2 for (idx, idy,idz) in zip(dx, dy,dz)]
            min_d = min(distances)
            nearest_idx = distances.index(min_d) + prev_idx  # Index of minimum distance
            ref_x, ref_y, ref_z = trajectory_bezier[corner_idx, nearest_idx]  # Get (x, y, z)
            if update_prev_idx:
                self.prev_waypoints_idx = nearest_idx 
            ref_points.append([ref_x,ref_y,ref_z])
        ref_points=np.array(ref_points)
        return ref_points
        
    
   
    # def _calc_epsilon(self, mu_y, sigma_y, mu_z, sigma_z, size_sample: int, size_time_step: int) -> np.ndarray:
    #     """Sample epsilon with two separate covariance matrices for x and z."""

    #     size_dim_u = 6  # 6 velocity inputs (x, y, z for 2 corners)
        
    #     if sigma_y.shape[0] != sigma_y.shape[1] or sigma_y.shape[0] != 2:
    #         raise ValueError("[ERROR] sigma_x must be a 2x2 square matrix for (corner1_x, corner2_x).")
    #     if sigma_z.shape[0] != sigma_z.shape[1] or sigma_z.shape[0] != 2:
    #         raise ValueError("[ERROR] sigma_z must be a 2x2 square matrix for (corner1_z, corner2_z).")

    #     epsilon_y = np.random.multivariate_normal(mu_y, sigma_y, (size_sample, size_time_step))
    #     epsilon_z = np.random.multivariate_normal(mu_z, sigma_z, (size_sample, size_time_step))
        
    #     # if self.count<8:
    #     #     epsilon_z= np.abs(epsilon_z)  # Make first half positive
    #     # else:
    #     #     epsilon_z= -np.abs(epsilon_z)  # Make second half negative

    #     epsilon = np.zeros((size_sample, size_time_step, size_dim_u))

    #     # Assign sampled values (no need for clipping)
    #     epsilon[:, :, 1] = epsilon_y[:, :, 0]  # Corner 1 y
    #     epsilon[:, :, 4] = np.abs(epsilon_y[:, :, 1])  # Corner 2 y
    #     epsilon[:, :, 0] = 0.0  # Corner 1 x
    #     epsilon[:, :, 3] = 0.0  # Corner 2 x
    #     epsilon[:, :, 2] = epsilon_z[:, :, 0]  # Corner 1 z
    #     epsilon[:, :, 5] = epsilon_z[:, :, 1]  # Corner 2 z
    #     window_size = 5  # Adjust this value for more or less smoothing
    #     #print(epsilon)
    #     for i in range(size_dim_u):
    #         epsilon[:, :, i] = scipy.ndimage.uniform_filter1d(epsilon[:, :, i], size=window_size, axis=1, mode='nearest')
    #     return epsilon

    def _calc_epsilon(self,noise_std,size_sample: int, size_time_step: int):
        size_dim_u = 6 
        perturbed_trajectories = np.zeros((size_sample,size_time_step,size_dim_u))
        # valid_steps = min(size_time_step, velocities.shape[0] - self.count)  
        for k in range(size_sample):
            noise1 = np.random.normal(0, noise_std)  # Single noise per trajectory
            noise2 = np.random.normal(0, noise_std)
            perturbed_trajectories[k] = velocities[self.count:self.count+size_time_step,:].copy()
            perturbed_trajectories[k, :, 0] += noise1  # Apply noise uniformly
            perturbed_trajectories[k, :, 2] += noise2
            perturbed_trajectories[k, :, 3] -= noise1
            perturbed_trajectories[k, :, 5] += noise2
        epsilon = np.zeros((size_sample, size_time_step, size_dim_u))
        epsilon = perturbed_trajectories
        window_size = 5
        size_time_step1 = min(size_time_step, epsilon.shape[1])
        for i in range(size_dim_u):
            epsilon[:, :, i] = scipy.ndimage.uniform_filter1d(epsilon[:, :, i], size=size_time_step1, axis=1, mode='nearest')
        return epsilon
    def calc_control_input(self, observed_x: np.ndarray):
        """
        Compute the optimal control input using Model Predictive Path Integral (MPPI).
        """
        # Load previous control input sequence
        u = self.u_prev
        x0 = observed_x  # Set initial x value from observation
        noise_std = 0.09

        # Check if cloth is close enough to reference configuration
        cloth_error = np.linalg.norm(x0 - x_ref)
        completion_threshold = 0.05  

        if cloth_error < completion_threshold:
            self.flag_done = True
            return (), [], [], [], self.flag_done

        # Buffer for cost evaluation
        S = np.zeros(self.K) 

        # Step 1: Sample noise epsilon
        epsilon = self._calc_epsilon(noise_std, self.K, self.T)

        # Step 2: Predict future states and evaluate cost
        v = np.zeros((self.K, self.T, self.dim_u))  
        x_init = cloth.get_particles() 
        for k in range(self.K):         
            x = x0 
            self.prev_waypoints_idx = self.count
            for t in range(1, self.T+1):
                # if k < (1.0 - self.param_exploration) * self.K:
                #     v[k, t-1] = u[t-1] + epsilon[k, t-1]  # Exploitation
                # else:
                #     v[k, t-1] = epsilon[k, t-1]  # Exploration
                print("time",t)
                v[k, t-1] = epsilon[k, t-1]
                v[k,t-1,1]=0.0
                v[k,t-1,4]=0.0
                if t<18:
                    v[k, t-1, 2] = np.clip(v[k, t-1, 2], 0.4, 1.5)   # Corner 1 z
                    v[k, t-1, 5] = np.clip(v[k, t-1, 5], 0.4, 1.5)   # Corner 2 z
                    v[k, t-1, 0] = np.clip(v[k, t-1, 0], -1.5, -0.8)  # Corner 1 y
                    v[k, t-1, 3] = np.clip(v[k, t-1, 3], -1.5, -0.8)  # Corner 2 y
                else:
                    v[k, t-1, 2] = np.clip(v[k, t-1, 2], -0.6, -0.5)   # Corner 1 z
                    v[k, t-1, 5] = np.clip(v[k, t-1, 5], -0.6, -0.5)   # Corner 2 z
                    v[k, t-1, 0] = np.clip(v[k, t-1, 0], -1.5, -1.2)  # Corner 1 y
                    v[k, t-1, 3] = np.clip(v[k, t-1, 3], -1.5, -1.2)  # Corner 2 y
                x = self._F(x, v[k, t-1])
                print("velocities",v[k,t-1])
                S[k] += self.compute_cloth_cost(x, v[k, t-1], u[t-1]) 
            S[k] += self.compute_cloth_cost(x ,0, 0)  # Terminal cost
            self.reset(x_init,x0,x) 

        # Step 3: Compute information-theoretic weights
        w = self._compute_weights(S)  
        w_epsilon = np.zeros((self.T, self.dim_u))  

        for t in range(self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # Apply moving average filter
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=self.T)
        u = w_epsilon
        u[:, 0] = np.clip(u[:, 0], -1.5, 1.5)  # Corner 1 y
        u[:, 3] = np.clip(u[:, 3], -1.5, 1.5)  # Corner 2 y
        u[:, 2] = np.clip(u[:, 2], 0.2, 1.5)   # Corner 1 z
        u[:, 5] = np.clip(u[:, 5], 0.2, 1.5)   # Corner 2 z
        # Update previous control input sequence (shift left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
        
        return u[0], u, self.flag_done
    
    def reset(self,x_init,x0,x):
            for i, pos in enumerate(x_init):
                pos = np.array(pos, dtype=np.float32)  # Ensure float32
                cloth.set_particle_position(i, pos)  # Convert to tuple
                cloth.set_particle_velocity(i,np.array([0.0,0.0,0.0]))
                cloth.release_particle(i)
            # for point in fixed_points:
            #     particle = cloth.find_closest_particle(point)
            #     cloth.fix_particle(particle)
            
    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """
        Predicts the next state of the cloth using Genesis' force application function.

        Args:
            x_t (np.ndarray): Current state [positions, velocities].
            v_t (np.ndarray): Control input (forces applied to grip points).
            dt (float): Time step for integration.

        Returns:
            np.ndarray: Predicted next state [positions, velocities].
        """
        x_t_plus_1 = x_t.copy()  # Initialize next state
        for step in range(80):
            for i in range(2):  # Iterate for selected particles
                particle_position = tuple(x_t[i])  
                particle = cloth.find_closest_particle(particle_position)   
                velocity = v_t[i * 3:(i + 1) * 3]  # Extract corresponding velocity
                #print("velocity",velocity)
                # velocity[0]=velocity[0]-0.5
                # velocity[2]=velocity[2]+0.5
                cloth.set_particle_velocity(particle, velocity)
                pos = cloth.get_particles()  
                x_t[i] = pos[particle]
            scene.step()  # Advance simulation

        for i in range(2):
             particle_position = tuple(x_t[i])  
             particle = cloth.find_closest_particle(particle_position)   
             velocity = np.array([0,0,0])  # Extract corresponding velocity
             cloth.set_particle_velocity(particle, velocity)
        for i in range(2):
            particle_pos = tuple(x_t[i])
            particle = cloth.find_closest_particle(particle_pos)
            pos = cloth.get_particles()
            x_t_plus_1[i] = pos[particle]
        return x_t_plus_1
        


    def _cost_jit(self, x_t: np.ndarray) -> float:
        """Calculate state cost using JIT-compiled function."""
        pass
    
    def compute_cloth_cost(self,x_t: np.ndarray, v_t: np.ndarray, prev_v_t: np.ndarray, 
                       w_tracking=1.0, w_control=0.1, w_smoothness=0.05) -> float:
        """
        Computes the cost function for cloth folding in MPPI.

        Args:
            x_t (np.ndarray): Current state [positions of key points].
            v_t (np.ndarray): Current control input (applied velocities).
            x_ref (np.ndarray): Reference positions for key points.
            prev_v_t (np.ndarray): Previous control input (to penalize abrupt changes).
            w_tracking (float): Weight for tracking error.
            w_control (float): Weight for minimizing control effort.
            w_smoothness (float): Weight for minimizing jerk.

        Returns:
            float: Cost value for this trajectory sample.
        """
        ref = self.get_nearest_reference_points(x_t,update_prev_idx=True)
        
        # Tracking cost: Sum of squared distances from the reference points
        J_tracking = np.sum(np.linalg.norm(x_t - ref, axis=1) ** 2)
        # Control effort cost: Sum of squared velocity inputs (penalizing excessive velocity)
        J_control = np.sum(v_t ** 2)

        # Smoothness cost: Sum of squared differences between successive velocity commands
        J_smoothness = np.sum((v_t - prev_v_t) ** 2)

        # Weighted sum of all cost terms
        total_cost = (w_tracking * J_tracking) + (w_control * J_control) + (w_smoothness * J_smoothness)

        return total_cost


    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """Compute weights for each sample."""
        # prepare buffer
        w = np.zeros((self.K))
        # calculate rho
        rho = S.min()
        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w
    
    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply moving average filter for smoothing input sequence.

        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean
    
    

# Let the cloth settle under gravity before applying control
settling_steps = 500  # Number of steps to wait before starting MPPI
print("Letting the cloth settle under gravity...")

for i in range(settling_steps):
    scene.step()  # Simulate one timestep with only gravity

# Initialize MPPI controller
horizon_step_T = 5 # Prediction horizon
u_prev = np.zeros((horizon_step_T, 6))  # Initialize previous control inputs

mppi = MPPI(u_prev, horizon_step_T)
optimal_control =[]
# Initial state of the cloth
observed_x = np.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0]])
 
for step in range(40):
    T = min(horizon_step_T, 40 - step)  # Adjust horizon based on remaining steps
    mppi.T = T  # Update MPPI with new horizon
    mppi.u_prev = np.zeros((T, 6))  # Reinitialize control input history
    optimal_u, _, flag_done = mppi.calc_control_input(observed_x)

    if flag_done:
        print(f"Cloth reached target configuration at step {step}")
        break
    
    # Apply the first control input from MPPI
    #print("giving the next control input")
    mppi.count = step
    observed_x = mppi._F(observed_x, optimal_u)  # Simulate next state
    optimal_control.append(optimal_u)
    print(f"Step {step}, Applied Control: {optimal_u}")
import csv

# Save the optimal_control list to a CSV file
with open("optimal_control1.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(optimal_control)

print("Optimal control saved to optimal_control.csv")

while True:
    scene.step()
