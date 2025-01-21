import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


import csv
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import json
import traceback

import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC, DQN

from DiscreteHybridEnv import DiscreteHybridEnv
from combined_pinn import CompetingHybridEnv


import sys
import os
import tempfile
import json
import traceback

import time
from datetime import timedelta


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
torch.set_num_threads(1)

from datetime import datetime  # Change this line

# ... existing code ...
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 

log_file = f"logs/training_log_{current_time}.txt"

# Create a custom logger class
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to both terminal and file
sys.stdout = Logger(log_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available
print("GPU Available: ", torch.cuda.is_available())

# Only try to get device name if CUDA is available
if torch.cuda.is_available():
    print("GPU Device Name: ", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using CPU")

# System Constants
NUM_BUSES = 33
NUM_EVCS = 5

# Base Values
S_BASE = 10e6      # VA
V_BASE_HV_AC = 12660  # V
V_BASE_LV_AC = 480   # V

V_BASE_DC = 800    # V
V_OUT_EVCS = 400
ATTACK_WEIGHT = 1.0

modulation_index_system= V_OUT_EVCS/V_BASE_DC

# Calculate base currents
I_BASE_HV_AC = S_BASE / (torch.sqrt(torch.tensor(3)) * V_BASE_HV_AC)  # HV AC base current
I_BASE_LV_AC = S_BASE / (torch.sqrt(torch.tensor(3)) * V_BASE_LV_AC)  # LV AC base current
I_BASE_DC = S_BASE / V_BASE_DC                       # DC base current

# Calculate base impedances
Z_BASE_HV_AC = V_BASE_HV_AC**2 / S_BASE              # HV AC base impedance
Z_BASE_LV_AC = V_BASE_LV_AC**2 / S_BASE              # LV AC base impedance
Z_BASE_DC = V_BASE_DC**2 / S_BASE                    # DC base impedance

# EVCS Parameters
EVCS_POWER = 50    # EVCS power rating in kW
EVCS_POWER_PU = EVCS_POWER * 1000 / S_BASE     # Convert kW to p.u.
EVCS_CAPACITY = EVCS_POWER * 1000 / S_BASE     # EVCS capacity in p.u. (same as power rating)
EVCS_EFFICIENCY = 0.98                         # EVCS conversion efficiency
EVCS_VOLTAGE = V_OUT_EVCS / V_BASE_DC          # Nominal voltage ratio

# Voltage Limits
MAX_VOLTAGE_PU = 1.05               # Maximum allowable voltage in per unit (reduced from 1.2)
MIN_VOLTAGE_PU = 0.95               # Minimum allowable voltage in per unit (increased from 0.8)
VOLTAGE_VIOLATION_PENALTY = 1000.0  # Penalty for voltage violations


V_OUT_NOMINAL = EVCS_VOLTAGE  # Nominal output voltage in p.u.
V_OUT_VARIATION = 0.05  # 5% allowed variation (reduced from 10%)

# Controller Parameters (in p.u.)
EVCS_PLL_KP = 0.1
EVCS_PLL_KI = 0.5
MAX_PLL_ERROR = 10.0

EVCS_OUTER_KP = 1
EVCS_OUTER_KI = 0.5

EVCS_INNER_KP = 1
EVCS_INNER_KI = 0.5
OMEGA_N = 2 * torch.pi * 60         # Nominal angular frequency (60 Hz)

# Wide Area Controller Parameters
# WAC_KP_VDC = 1.0
# WAC_KI_VDC = 0.5

WAC_KP_VDC =[1, 1, 1, 1, 1]
WAC_KI_VDC =[0.5, 0.5, 0.5, 0.5, 0.5]

WAC_KP_VDC = torch.tensor(WAC_KP_VDC)
WAC_KI_VDC = torch.tensor(WAC_KI_VDC)




WAC_KP_VOUT = 1.0
WAC_KI_VOUT = 0.5

# WAC_KP_VOUT =[1, 1, 1, 1, 1]
# WAC_KI_VOUT =[0.5, 0.5, 0.5, 0.5, 0.5]

WAC_DC_LINK_VOLTAGE_SETPOINT = V_BASE_DC / V_BASE_DC  # Desired DC voltage in p.u.
WAC_VOUT_SETPOINT = V_OUT_EVCS / V_BASE_DC     # Desired output voltage in p.u.

# Circuit Parameters (convert to p.u.)
CONSTRAINT_WEIGHT = 1.0
LCL_L1 = 0.002 / Z_BASE_LV_AC     # LCL filter inductor 1
LCL_L2 = 0.002 / Z_BASE_LV_AC     # LCL filter inductor 2
LCL_CF = 10e-1 * S_BASE / (V_BASE_LV_AC**2)  # LCL filter capacitor
R = 0.1 / Z_BASE_LV_AC           # Resistance
C_dc = 0.01 * S_BASE / (V_BASE_DC**2)  # DC-link capacitor (modified to use DC base)
L_dc = 0.001 / Z_BASE_DC      # DC inductor (modified to use DC base)
v_battery = 800 / V_BASE_DC   # Battery voltage in p.u.
R_battery = 0.1 / Z_BASE_DC   # Battery resistance (modified to use DC base)

# Time parameters
TIME_STEP = 0.1  # 1 ms
TOTAL_TIME = 500  # 100 seconds

POWER_BALANCE_WEIGHT = 1.0
RATE_OF_CHANGE_LIMIT = 0.05  # Maximum 5% change per time step
VOLTAGE_STABILITY_WEIGHT = 2.0
POWER_FLOW_WEIGHT = 1.5
MIN_VOLTAGE_LIMIT = 0.85  # Minimum allowable voltage
THERMAL_LIMIT_WEIGHT = 1.0
COORDINATION_WEIGHT = 0.8  # Weight for coordinated attack impact



EVCS_BUSES = [2, 4, 8, 25, 30]           # Location of EVCS units

# Load IEEE 33-bus system data
line_data = [
    (1, 2, 0.0922, 0.0477), (2, 3, 0.493, 0.2511), (3, 4, 0.366, 0.1864), (4, 5, 0.3811, 0.1941),
    (5, 6, 0.819, 0.707), (6, 7, 0.1872, 0.6188), (7, 8, 1.7114, 1.2351), (8, 9, 1.03, 0.74),
    (9, 10, 1.04, 0.74), (10, 11, 0.1966, 0.065), (11, 12, 0.3744, 0.1238), (12, 13, 1.468, 1.155),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.591, 0.526), (15, 16, 0.7463, 0.545), (16, 17, 1.289, 1.721),
    (17, 18, 0.732, 0.574), (2, 19, 0.164, 0.1565), (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373), (3, 23, 0.4512, 0.3083), (23, 24, 0.898, 0.7091), (24, 25, 0.896, 0.7011),
    (6, 26, 0.203, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.059, 0.9337), (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.963), (31, 32, 0.31, 0.3619), (32, 33, 0.341, 0.5302)
]

bus_data = np.array([
    [1, 0, 0, 0], [2, 100, 60, 0], [3, 70, 40, 0], [4, 120, 80, 0], [5, 80, 30, 0],
    [6, 60, 20, 0], [7, 145, 100, 0], [8, 160, 100, 0], [9, 60, 20, 0], [10, 60, 20, 0],
    [11, 100, 30, 0], [12, 60, 35, 0], [13, 60, 35, 0], [14, 80, 80, 0], [15, 100, 10, 0],
    [16, 100, 20, 0], [17, 60, 20, 0], [18, 90, 40, 0], [19, 90, 40, 0], [20, 90, 40, 0],
    [21, 90, 40, 0], [22, 90, 40, 0], [23, 90, 40, 0], [24, 420, 200, 0], [25, 380, 200, 0],
    [26, 100, 25, 0], [27, 60, 25, 0], [28, 60, 20, 0], [29, 120, 70, 0], [30, 200, 600, 0],
    [31, 150, 70, 0], [32, 210, 100, 0], [33, 60, 40, 0]
])

line_data = torch.tensor(line_data, dtype=torch.float32)
bus_data = torch.tensor(bus_data, dtype=torch.float32)

# Convert bus data to per-unit
bus_data[:, 1:3] = bus_data[:, 1:3] * 1e3 / S_BASE

# Initialize Y-bus matrix
Y_bus = torch.zeros((NUM_BUSES, NUM_BUSES), dtype=torch.complex64)



# Fill Y-bus matrix
for line in line_data:
    from_bus, to_bus, r, x = line
    from_bus, to_bus = int(from_bus)-1 , int(to_bus)-1 # Convert to 0-based index
    y = 1/complex(r, x)
    Y_bus[from_bus, from_bus] += y
    Y_bus[to_bus, to_bus] += y
    Y_bus[from_bus, to_bus] -= y
    Y_bus[to_bus, from_bus] -= y

# Convert to TensorFlow constant
if isinstance(Y_bus, torch.Tensor):
    Y_bus_tf = Y_bus.clone().detach().to(dtype=torch.complex64)
else:
    Y_bus_tf = torch.tensor(Y_bus, dtype=torch.complex64)


G_d = None
G_q = None

def initialize_conductance_matrices():
    """Initialize conductance matrices from Y-bus matrix"""
    global G_d, G_q, B_d, B_q
    # Extract G (conductance) and B (susceptance) matrices
    G_d = torch.real(Y_bus_tf)  # Real part for d-axis
    G_q = torch.real(Y_bus_tf)  # Real part for q-axis
    B_d = torch.imag(Y_bus_tf)  # Imaginary part for d-axis
    B_q = torch.imag(Y_bus_tf)  # Imaginary part for q-axis
    return G_d, G_q, B_d, B_q

# Call this function before training starts
G_d, G_q, B_d, B_q = initialize_conductance_matrices()

# For individual elements (if needed)
G_d_kh = torch.diag(G_d)  # Diagonal elements for d-axis conductance
G_q_kh = torch.diag(G_q)  # Diagonal elements for q-axis conductance
B_d_kh = torch.diag(B_d)  # Diagonal elements for d-axis susceptance
B_q_kh = torch.diag(B_q)  # Diagonal elements for q-axis susceptance



class SACWrapper(gym.Env):
    def __init__(self, env, agent_type, dqn_agent=None, sac_defender=None, sac_attacker=None):
        super(SACWrapper, self).__init__()
        
        self.env = env
        self.agent_type = agent_type
        self.dqn_agent = dqn_agent
        self.sac_defender = sac_defender
        self.sac_attacker = sac_attacker
        self.NUM_EVCS = env.NUM_EVCS
        self.TIME_STEP = env.TIME_STEP  # Add this line to fix the missing attribute
        
        # Initialize tracking variables
        self.voltage_deviations = torch.zeros(self.NUM_EVCS, dtype=torch.float32)
        self.cumulative_deviation = 0.0
        self.attack_active = False
        self.target_evcs = torch.zeros(self.NUM_EVCS)
        self.attack_duration = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rewards = 0.0
        
        # Define observation and action spaces using numpy dtypes
        self.observation_space = env.observation_space
        if agent_type == 'attacker':
            self.action_space = gym.spaces.Box(
                low=np.float32(-1.0),  # Changed from torch.tensor to np.float32
                high=np.float32(1.0),  # Changed from torch.tensor to np.float32
                shape=(self.NUM_EVCS * 2,),
                dtype=np.float32  # Changed from torch.float32 to np.float32
            )
        else:  # defender
            self.action_space = gym.spaces.Box(
                low=np.float32(0.0),  # Changed from torch.tensor to np.float32
                high=np.float32(1.0),  # Changed from torch.tensor to np.float32
                shape=(self.NUM_EVCS * 2,),
                dtype=np.float32  # Changed from torch.float32 to np.float32
            )
        
        # Initialize state
        self.state = None
        self.reset()

    def step(self, action):
        try:
            # Convert input action to proper shape tensor
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif isinstance(action, list):
                action = torch.tensor(action, dtype=torch.float32)
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor([action], dtype=torch.float32)
            
            # Get DQN action
            dqn_state = torch.as_tensor(self.state, dtype=torch.float32).reshape(1, -1)
            dqn_raw = self.dqn_agent.predict(dqn_state.numpy(), deterministic=True)
            dqn_action = torch.tensor(dqn_raw[0] if isinstance(dqn_raw, tuple) else dqn_raw, dtype=torch.int64)
            
            # Process actions based on agent type
            if self.agent_type == 'attacker':
                attacker_action = action
                if self.sac_defender is not None:
                    defender_state = torch.as_tensor(self.state, dtype=torch.float32)
                    defender_raw = self.sac_defender.predict(defender_state.numpy(), deterministic=True)
                    defender_action = torch.tensor(defender_raw[0], dtype=torch.float32)
                else:
                    defender_action = torch.zeros(self.NUM_EVCS * 2, dtype=torch.float32)
            else:  # defender
                defender_action = action
                if self.sac_attacker is not None:
                    attacker_state = torch.as_tensor(self.state, dtype=torch.float32)
                    attacker_raw = self.sac_attacker.predict(attacker_state.numpy(), deterministic=True)
                    attacker_action = torch.tensor(attacker_raw[0], dtype=torch.float32)
                else:
                    attacker_action = torch.zeros(self.NUM_EVCS * 2, dtype=torch.float32)
            
            # Combine actions into dictionary
            combined_action = {
                'dqn': dqn_action.numpy(),
                'attacker': attacker_action.numpy(),
                'defender': defender_action.numpy()
            }
            
            # Take step in environment
            next_state, rewards, done, truncated, info = self.env.step(combined_action)

            # Handle rewards based on agent type
            if isinstance(rewards, dict):
                # Extract reward for current agent type
                reward = float(rewards.get(self.agent_type, 0.0))
            elif isinstance(rewards, (int, float)):
                reward = float(rewards)
            else:
                reward = 0.0  # Default reward if invalid type
                print(f"Warning: Unexpected reward type: {type(rewards)}")

            # Convert state to numpy array for return
            if isinstance(next_state, torch.Tensor):
                state_np = next_state.detach().numpy()
            else:
                state_np = np.array(next_state, dtype=np.float32)

            return state_np, reward, bool(done), bool(truncated), dict(info)

        except Exception as e:
            print(f"Error in SACWrapper step: {str(e)}")
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,  # End episode on error
                False,
                {'error': str(e)}
            )

    def reset(self, seed=None, options=None):
        try:
            # Reset tracking variables
            self.voltage_deviations = torch.zeros(self.NUM_EVCS, dtype=torch.float32)
            self.cumulative_deviation = 0.0
            self.attack_active = False
            self.target_evcs = torch.zeros(self.NUM_EVCS)
            self.attack_duration = 0.0
            self.rewards = 0.0
            # Reset environment
            obs_info = self.env.reset(seed=seed)
            
            # Handle different return types
            if isinstance(obs_info, tuple):
                obs, info = obs_info
            else:
                obs = obs_info
                info = {}
            
            # Convert observation to proper format
            if isinstance(obs, np.ndarray):
                self.state = torch.from_numpy(obs).float()
            else:
                self.state = torch.tensor(obs, dtype=torch.float32)
            
            # Return numpy array for SB3 compatibility
            return self.state.numpy(), dict(info)
            
        except Exception as e:
            print(f"Error in SACWrapper reset: {str(e)}")
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                {'error': str(e)}
            )

    def update_agents(self, dqn_agent= None, sac_defender=None, sac_attacker= None):
        """Update the agents used by the wrapper."""
        if dqn_agent is not None:
            self.dqn_agent = dqn_agent
            print("Updated DQN agent")
        if sac_defender is not None:
            self.sac_defender = sac_defender
            print("Updated SAC defender")
        if sac_attacker is not None:
            self.sac_attacker = sac_attacker
            print("Updated SAC attacker")

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()


class EVCS_PowerSystem_PINN(nn.Module):
    def __init__(self, num_buses=NUM_BUSES, num_evcs=NUM_EVCS):
        super(EVCS_PowerSystem_PINN, self).__init__()
        
        self.num_buses = num_buses
        self.num_evcs = num_evcs
        
        # Dense layer
        self.dense1 = nn.Linear(1, 256)  # Input is time (1-dimensional)
        self.tanh = nn.Tanh()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm3 = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm4 = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(512, num_buses * 3 + num_evcs * 18)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Initialize with a dummy input
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def get_state(self, t):
        """Extract state information from model outputs"""
        outputs = self.forward(t)
        
        # Extract components
        v_d = outputs[:, :NUM_BUSES]
        v_q = outputs[:, NUM_BUSES:2*NUM_BUSES]
        evcs_vars = outputs[:, 2*NUM_BUSES:]
        
        # Create state vector
        state = torch.cat([
            v_d,  # Voltage d-axis components
            v_q,  # Voltage q-axis components
            torch.sqrt(v_d**2 + v_q**2),  # Voltage magnitudes
            evcs_vars  # EVCS-specific variables
        ], dim=1)
        
        return state
    
    def forward(self, t):
        # Ensure input is on correct device and proper shape
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        
        t = t.to(self.device)
        
        # Reshape if necessary
        if len(t.shape) == 1:
            t = t.unsqueeze(0)  # Add batch dimension
        
        # Initial dense transformation
        x = self.tanh(self.dense1(t))
        
        # Reshape for LSTM (batch_size, timesteps, features)
        x = x.unsqueeze(1)  # Add time dimension
        
        # LSTM processing
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, (h_n, _) = self.lstm4(x)
        
        # Use the last hidden state
        x = h_n[-1]
        
        # Output layer
        output = self.output_layer(x)
        
        # Split and process outputs
        voltage_magnitude = torch.exp(output[:, :NUM_BUSES])  # Ensure positive
        voltage_angle = torch.atan(output[:, NUM_BUSES:2*NUM_BUSES])  # Bound angles
        evcs_outputs = torch.tanh(output[:, 2*NUM_BUSES:])  # Bound EVCS outputs
        
        # Concatenate outputs
        return torch.cat([voltage_magnitude, voltage_angle, evcs_outputs], dim=1)
    
    @property
    def trainable_parameters(self):
        """Get all trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    

class SafeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for backward pass
        ctx.save_for_backward(x)
        # Return x where finite, small value where not
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x) + 1e-30)

    @staticmethod
    def backward(ctx, grad_output):
        # Get saved input
        x, = ctx.saved_tensors
        # Return gradient where finite, zero where not
        return torch.where(torch.isfinite(grad_output), grad_output, torch.zeros_like(grad_output) + 1e-30)

def safe_op(x):
    """Safely perform tensor operations with proper gradient handling."""
    return SafeOp.apply(x)



def safe_matrix_operations(func):
    """Decorator for safe matrix operations with logging"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Handle tuple return type properly
            if isinstance(result, tuple):
                nan_check = any(torch.isnan(r).any() for r in result if isinstance(r, torch.Tensor))
                if nan_check:
                    print(f"Warning: NaN detected in {func.__name__}")
                    print(f"Input shapes: {[arg.shape if isinstance(arg, torch.Tensor) else None for arg in args]}")
                    return tuple(torch.zeros_like(r) if isinstance(r, torch.Tensor) else r for r in result)
                return result
            else:
                if torch.isnan(result).any():
                    print(f"Warning: NaN detected in {func.__name__}")
                    print(f"Input shapes: {[arg.shape if isinstance(arg, torch.Tensor) else None for arg in args]}")
                    return torch.zeros_like(result)
                return result
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            # Get shapes from first argument assuming it's a tensor
            if isinstance(args[0], torch.Tensor):
                batch_size = args[0].shape[0]
                num_buses = args[0].shape[1]
                return (torch.zeros(batch_size, num_buses), 
                       torch.zeros(batch_size, num_buses), 
                       {})
            else:
                raise ValueError("First argument must be a tensor")
    return wrapper


def calculate_power_flow_base(v_d, v_q, G, B, bus_mask):
    """Base power flow calculation with proper shape handling."""
    try:
        # Ensure inputs are tensors
        if not isinstance(v_d, torch.Tensor):
            v_d = torch.tensor(v_d, dtype=torch.float32)
        if not isinstance(v_q, torch.Tensor):
            v_q = torch.tensor(v_q, dtype=torch.float32)
            
        # Ensure inputs are rank 2 [batch_size, num_buses]
        v_d = v_d.reshape(-1, v_d.shape[-1])  # [batch, buses]
        v_q = v_q.reshape(-1, v_q.shape[-1])  # [batch, buses]
        
        # Matrix multiplication for power calculations
        # P = V_d * (G * V_d + B * V_q) + V_q * (G * V_q - B * V_d)
        G_vd = torch.matmul(G, v_d.unsqueeze(-1))  # Changed from expand_dims
        G_vq = torch.matmul(G, v_q.unsqueeze(-1))  # Changed from expand_dims
        B_vd = torch.matmul(B, v_d.unsqueeze(-1))  # Changed from expand_dims
        B_vq = torch.matmul(B, v_q.unsqueeze(-1))  # Changed from expand_dims
        
        # Calculate P and Q
        P = v_d * torch.squeeze(G_vd, -1) + v_q * torch.squeeze(G_vq, -1)  # [batch, buses]
        Q = v_d * torch.squeeze(B_vd, -1) - v_q * torch.squeeze(B_vq, -1)  # [batch, buses]
        
        # Apply mask
        P = P * bus_mask  # [batch, buses]
        Q = Q * bus_mask  # [batch, buses]
        
        return P, Q
        
    except Exception as e:
        print("\nERROR in calculate_power_flow_base:")       
        print(str(e))
        print("Error type:", type(e).__name__)
        return None, None, {}

def calculate_power_flow_pcc(v_d, v_q, G, B):
    """PCC power flow calculation."""
    num_buses = v_d.shape[-1]
    # Convert lists to tensors before concatenation
    mask = torch.cat([torch.tensor([1.0]), torch.zeros(num_buses - 1)])  # Changed from concat to cat
    mask = mask.unsqueeze(0)  # Changed from expand_dims to unsqueeze
    return calculate_power_flow_base(v_d, v_q, G, B, mask)

def calculate_power_flow_load(v_d, v_q, G, B):
    """Load bus power flow calculation."""
    num_buses = v_d.shape[-1]
    mask = torch.ones(1, num_buses)  # Direct tensor creation
    # Use index assignment instead of tensor_scatter_nd_update
    mask[0, 0] = 0.0  # Zero out PCC bus
    # Zero out EVCS buses
    for bus in EVCS_BUSES:
        mask[0, bus] = 0.0
    return calculate_power_flow_base(v_d, v_q, G, B, mask)

def calculate_power_flow_ev(v_d, v_q, G, B):
    """EV bus power flow calculation."""
    num_buses = v_d.shape[-1]
    mask = torch.zeros(1, num_buses)  # Direct tensor creation
    # Set EVCS buses to 1
    for bus in EVCS_BUSES:
        mask[0, bus] = 1.0
    return calculate_power_flow_base(v_d, v_q, G, B, mask)


def physics_loss(model, t, Y_bus_tf, bus_data, attack_actions, defend_actions):
    """Calculate physics-based losses with proper gradient handling."""
    try:
        # Initialize all loss components to zero
        power_flow_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        evcs_total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        wac_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        V_regulation_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        # Convert Y_bus_tf properly
        WAC_KP_VOUT = 1.0
        WAC_KI_VOUT = 0.5
        if isinstance(Y_bus_tf, torch.Tensor):
            Y_bus_tf = Y_bus_tf.clone().detach().to(dtype=torch.complex64)
        else:
            Y_bus_tf = torch.tensor(Y_bus_tf, dtype=torch.complex64)

        # Convert inputs properly
        if isinstance(t, torch.Tensor):
            t = t.clone().detach().reshape(-1, 1)
        else:
            t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

        # Handle attack/defend actions
        if isinstance(attack_actions, torch.Tensor):
            attack_actions = attack_actions.clone().detach().requires_grad_(True)
        else:
            attack_actions = torch.tensor(attack_actions, dtype=torch.float32, requires_grad=True)

        if isinstance(defend_actions, torch.Tensor):
            defend_actions = defend_actions.clone().detach().requires_grad_(True)
        else:
            defend_actions = torch.tensor(defend_actions, dtype=torch.float32, requires_grad=True)

        # Initialize variables
        wac_error_vout = torch.zeros_like(t)
        wac_integral_vout = torch.zeros_like(t)
        
        # Fix WAC control calculation
        wac_control = (WAC_KP_VOUT * wac_error_vout + WAC_KI_VOUT * wac_integral_vout).clone().detach()
        modulation_index_vout = safe_op(torch.clamp(wac_control, min=0.0, max=1.0))

        # Ensure Y_bus_tf is complex
        if not Y_bus_tf.is_complex():
            Y_bus_tf = Y_bus_tf.to(dtype=torch.complex64)
    
        # Convert inputs to proper tensors
        t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
        attack_actions = torch.tensor(attack_actions, dtype=torch.float32, requires_grad=True)
        defend_actions = torch.tensor(defend_actions, dtype=torch.float32, requires_grad=True)
        
        # Extract real and imaginary parts of Y_bus
        G = Y_bus_tf.real.to(torch.float32)
        B = Y_bus_tf.imag.to(torch.float32)
        
        # Initialize loss components
        evcs_loss = []
        attack_loss = torch.tensor(0.0, dtype=torch.float32)
        voltage_violation_loss = torch.tensor(0.0, dtype=torch.float32)
        voltage_regulation_loss = torch.tensor(0.0, dtype=torch.float32)
        modulation_index_vout = torch.tensor(0.0, dtype=torch.float32)
        
        # Extract attack and defense actions
        fdi_voltage = attack_actions[:, :NUM_EVCS].reshape(-1, NUM_EVCS)
        fdi_current_d = attack_actions[:, NUM_EVCS:].reshape(-1, NUM_EVCS)
        KP_VOUT = defend_actions[:, :NUM_EVCS].reshape(-1, NUM_EVCS)
        KI_VOUT = defend_actions[:, NUM_EVCS:].reshape(-1, NUM_EVCS)

        with torch.enable_grad():
            # Get predictions and ensure proper shapes
            predictions = model(t)  # [batch_size, output_dim]
            
            # Extract predictions with explicit shapes
            V = safe_op(torch.exp(predictions[:, :NUM_BUSES]))  # [batch_size, NUM_BUSES]
            I = safe_op(predictions[:, NUM_BUSES:2*NUM_BUSES])  # [batch_size, NUM_BUSES]
            theta = safe_op(torch.atan(predictions[:, 2*NUM_BUSES:3*NUM_BUSES]))  # [batch_size, NUM_BUSES]
            evcs_vars = predictions[:, 3*NUM_BUSES:]
            
            # Calculate voltage components
            v_d = V * torch.cos(theta)
            v_q = V * torch.sin(theta)
            
            # Calculate power flows
            P_g_pcc, Q_g_pcc = calculate_power_flow_pcc(v_d, v_q, G, B)
            P_g_load, Q_g_load = calculate_power_flow_load(v_d, v_q, G, B)
            P_g_ev_load, Q_g_ev_load = calculate_power_flow_ev(v_d, v_q, G, B)
            
            # Calculate power mismatches
            P_mismatch = P_g_pcc - (P_g_load + P_g_ev_load)
            Q_mismatch = Q_g_pcc - (Q_g_load + Q_g_ev_load)
            
            # Calculate power flow loss
            power_flow_loss = safe_op(torch.mean(torch.square(P_mismatch) + torch.square(Q_mismatch)))
            
            # Initialize EVCS losses list and WAC variables
            evcs_loss = []
            wac_error_vdc = torch.zeros_like(t)
            wac_integral_vdc = torch.zeros_like(t)
            wac_error_vout = torch.zeros_like(t)
            wac_integral_vout = torch.zeros_like(t)
            
            # Process each EVCS with proper indexing
            for i, bus in enumerate(EVCS_BUSES):
                try:
                    # Ensure evcs_vars has proper batch dimension
                    evcs = evcs_vars[:, i*18:(i+1)*18]  # Shape should be [batch_size, 18]
                    
                    # Instead of using split, directly index the tensor
                    v_ac  = evcs[:, 0:1]
                    i_ac  = evcs[:, 1:2]
                    v_dc  = evcs[:, 2:3]
                    i_dc  = evcs[:, 3:4]
                    v_out = evcs[:, 4:5]
                    i_out = evcs[:, 5:6]
                    i_L1  = evcs[:, 6:7]
                    i_L2  = evcs[:, 7:8]
                    v_c = evcs[:, 8:9]
                    soc = evcs[:, 9:10]
                    delta = evcs[:, 10:11]
                    omega = evcs[:, 11:12]
                    phi_d = evcs[:, 12:13]
                    phi_q = evcs[:, 13:14]
                    gamma_d = evcs[:, 14:15]
                    gamma_q = evcs[:, 15:16]
                    i_d = evcs[:, 16:17]
                    i_q = evcs[:, 17:18]

                    # Clarke and Park Transformations
                    v_ac = safe_op(v_ac)
                    i_ac = safe_op(i_ac)
                    v_dc = safe_op(v_dc)
                    i_dc = safe_op(i_dc)
                    v_out = safe_op(v_out)
                    i_out = safe_op(i_out)
                    i_L1 = safe_op(i_L1)
                    i_L2 = safe_op(i_L2)
                    v_c = safe_op(v_c)

                    v_alpha = v_ac
                    v_beta = torch.zeros_like(v_ac)
                    i_alpha = i_ac
                    i_beta = torch.zeros_like(i_ac)

                    # Apply FDI attack
                    v_out_attacked = v_out + fdi_voltage[:, i:i+1]
                    i_d = i_d + fdi_current_d[:, i:i+1]
                    
                    v_d_evcs = safe_op(v_alpha * torch.cos(delta) + v_beta * torch.sin(delta))
                    v_q_evcs = safe_op(-v_alpha * torch.sin(delta) + v_beta * torch.cos(delta))
                    i_d_measured = safe_op(i_alpha * torch.cos(delta) + i_beta * torch.sin(delta))
                    i_q_measured = safe_op(-i_alpha * torch.sin(delta) + i_beta * torch.cos(delta))
                    
                    # PLL Dynamics
                    v_q_normalized = torch.tanh(safe_op(v_q_evcs))
                    pll_error = safe_op(EVCS_PLL_KP * v_q_normalized + EVCS_PLL_KI * phi_q)
                    pll_error = torch.clamp(torch.tensor(pll_error, dtype=torch.float32), torch.tensor(-MAX_PLL_ERROR, dtype=torch.float32), torch.tensor(MAX_PLL_ERROR, dtype=torch.float32))

                    wac_error_vdc += WAC_DC_LINK_VOLTAGE_SETPOINT - v_dc

                    wac_integral_vdc += wac_error_vdc * TIME_STEP
                    wac_output_vdc = WAC_KP_VDC[i]* wac_error_vdc + WAC_KI_VDC[i] * wac_integral_vdc

                    v_dc_ref = WAC_DC_LINK_VOLTAGE_SETPOINT + wac_output_vdc

                    v_out_ref = safe_op(v_dc_ref * (modulation_index_system))
                    
                    # Converter Outer Loop
                    i_d_ref = safe_op(EVCS_OUTER_KP * (v_dc - v_dc_ref) + EVCS_OUTER_KI * gamma_d)
                    i_q_ref = safe_op(EVCS_OUTER_KP * (0 - v_q) + EVCS_OUTER_KI * gamma_q)

                    # Converter Inner Loop
                    v_d_conv = safe_op(EVCS_INNER_KP * (i_d_ref - i_d) + EVCS_INNER_KI * phi_d - omega * LCL_L1 * i_q + v_d)
                    v_q_conv = safe_op(EVCS_INNER_KP * (i_q_ref - i_q) + EVCS_INNER_KI * phi_q + omega * LCL_L1 * i_d + v_q)

                    WAC_KI_VOUT = KI_VOUT[:,i:i+1]
                    WAC_KP_VOUT = KP_VOUT[:,i:i+1]

                    wac_error_vout += v_out_ref - v_out_attacked
                    wac_integral_vout += wac_error_vout * TIME_STEP

                    modulation_index_vout = safe_op(torch.clamp(
                        torch.tensor(WAC_KP_VOUT * wac_error_vout + WAC_KI_VOUT * wac_integral_vout, dtype=torch.float32), 
                        min=torch.tensor(0.0, dtype=torch.float32), 
                        max=torch.tensor(1.0, dtype=torch.float32)
                    ))

                    v_out = safe_op(v_dc * (modulation_index_vout + modulation_index_system))
                    
                    v_out_loss = safe_op(torch.mean(torch.square(v_out_ref - v_out)))

                    v_out_lower = V_OUT_NOMINAL * (1 - V_OUT_VARIATION)
                    v_out_upper = V_OUT_NOMINAL * (1 + V_OUT_VARIATION)
                    zero_tensor = torch.zeros_like(v_out)
                    v_out_constraint = safe_op(torch.mean(torch.square(
                        torch.maximum(zero_tensor, v_out_lower - v_out) + 
                        torch.maximum(zero_tensor, v_out - v_out_upper)
                    )))

                    # Calculate voltage deviation from nominal
                    voltage_deviation = torch.abs(v_out_ref- v_out)
                    impact_scale = torch.exp(voltage_deviation * 2.0) 

                    attack_loss += -ATTACK_WEIGHT * torch.mean(
                    v_out_loss * impact_scale
                    ) # Penalize attack magnitude

                    # Penalize voltages outside acceptable range
                    zero_tensor = torch.zeros_like(v_out)
                    upper_violation = torch.maximum(zero_tensor, v_out - torch.tensor(MAX_VOLTAGE_PU))
                    lower_violation = torch.maximum(zero_tensor, torch.tensor(MIN_VOLTAGE_PU) - v_out)

                    voltage_violation_loss += torch.mean(
                        VOLTAGE_VIOLATION_PENALTY * (torch.square(upper_violation) + torch.square(lower_violation))
                    )

                    zero_tensor = torch.zeros_like(v_out)
                    v_out_regulation_loss = safe_op(torch.mean(
                        torch.square(torch.maximum(zero_tensor, torch.tensor(0.95) - v_out)) + 
                        torch.square(torch.maximum(zero_tensor, v_out - torch.tensor(1.05)))
                    ))      

                    VOLTAGE_REG_WEIGHT = 10.0 # Weight for voltage regulation

                    voltage_regulation_loss += VOLTAGE_REG_WEIGHT * v_out_regulation_loss

                    # # Make sure the tensors have requires_grad=True
                    # delta = delta.requires_grad_(True)
                    # omega = omega.requires_grad_(True)
                    # phi_d = phi_d.requires_grad_(True)
                    # phi_q = phi_q.requires_grad_(True)
                    # i_d = i_d.requires_grad_(True)
                    # i_q = i_q.requires_grad_(True)
                    # i_L1 = i_L1.requires_grad_(True)
                    # i_L2 = i_L2.requires_grad_(True)
                    # v_c = v_c.requires_grad_(True)
                    # v_dc = v_dc.requires_grad_(True)
                    # i_out = i_out.requires_grad_(True)
                    
                    # Calculate losses
                    ddelta_dt = calculate_gradient(delta, TIME_STEP)
                    domega_dt = calculate_gradient(omega, TIME_STEP)
                    dphi_d_dt = calculate_gradient(phi_d, TIME_STEP)
                    dphi_q_dt = calculate_gradient(phi_q, TIME_STEP)
                    di_d_dt = calculate_gradient(i_d, TIME_STEP)
                    di_q_dt = calculate_gradient(i_q, TIME_STEP)
                    di_L1_dt = calculate_gradient(i_L1, TIME_STEP)
                    di_L2_dt = calculate_gradient(i_L2, TIME_STEP)
                    dv_c_dt = calculate_gradient(v_c, TIME_STEP)
                    dv_dc_dt = calculate_gradient(v_dc, TIME_STEP)
                    di_out_dt = calculate_gradient(i_out, TIME_STEP)


                    # ddelta_dt = delta.grad / TIME_STEP if delta.grad is not None else torch.zeros_like(delta)
                    # domega_dt = omega.grad / TIME_STEP if omega.grad is not None else torch.zeros_like(omega)
                    # dphi_d_dt = phi_d.grad / TIME_STEP if phi_d.grad is not None else torch.zeros_like(phi_d)
                    # dphi_q_dt = phi_q.grad / TIME_STEP if phi_q.grad is not None else torch.zeros_like(phi_q)
                    # di_d_dt = i_d.grad / TIME_STEP if i_d.grad is not None else torch.zeros_like(i_d)
                    # di_q_dt = i_q.grad / TIME_STEP if i_q.grad is not None else torch.zeros_like(i_q)
                    # di_L1_dt = i_L1.grad / TIME_STEP if i_L1.grad is not None else torch.zeros_like(i_L1)
                    # di_L2_dt = i_L2.grad / TIME_STEP if i_L2.grad is not None else torch.zeros_like(i_L2)
                    # dv_c_dt = v_c.grad / TIME_STEP if v_c.grad is not None else torch.zeros_like(v_c)
                    # dv_dc_dt = v_dc.grad / TIME_STEP if v_dc.grad is not None else torch.zeros_like(v_dc)
                    # di_out_dt = i_out.grad / TIME_STEP if i_out.grad is not None else torch.zeros_like(i_out)



                    P_ac = safe_op(v_d_evcs * i_d + v_q_evcs * i_q)

                    v_out_lower = V_OUT_NOMINAL * (1 - V_OUT_VARIATION)
                    v_out_upper = V_OUT_NOMINAL * (1 + V_OUT_VARIATION)
                    zero_tensor = torch.zeros_like(v_out)
                    v_out_constraint = safe_op(torch.mean(torch.square(
                        torch.maximum(zero_tensor, v_out_lower - v_out) + 
                        torch.maximum(zero_tensor, v_out - v_out_upper)
                    )))

                    P_dc = safe_op(v_dc * i_dc)
                    P_out = safe_op(v_out * i_out)
                    DC_DC_EFFICIENCY = 0.98

                    di_d_dt_loss = safe_op(torch.mean(torch.square(di_d_dt - (1/LCL_L1) * (v_d_conv - R * i_d - v_d + omega * LCL_L1 * i_q))))
                    di_q_dt_loss = safe_op(torch.mean(torch.square(di_q_dt - (1/LCL_L1) * (v_q_conv - R * i_q - v_q - omega * LCL_L1 * i_d))))

                    di_L1_dt_loss = safe_op(torch.mean(torch.square(di_L1_dt - (1/LCL_L1) * (v_d_conv - v_c - R * i_L1))))
                    di_L2_dt_loss = safe_op(torch.mean(torch.square(di_L2_dt - (1/LCL_L2) * (v_c - v_ac - R * i_L2))))
                    dv_c_dt_loss = safe_op(torch.mean(torch.square(dv_c_dt - (1/LCL_CF) * (i_L1 - i_L2))))

                    # Calculate EVCS losses with safe handling
                    evcs_losses = [
                        safe_op(torch.mean(torch.square(ddelta_dt - omega))),
                        safe_op(torch.mean(torch.square(domega_dt - pll_error))),
                        safe_op(torch.mean(torch.square(dphi_d_dt - v_d_evcs))),
                        safe_op(torch.mean(torch.square(dphi_q_dt - v_q_evcs))),
                        safe_op(torch.mean(torch.square(di_d_dt_loss))),
                        safe_op(torch.mean(torch.square(di_q_dt_loss))),
                        safe_op(torch.mean(torch.square(di_L1_dt_loss))),
                        safe_op(torch.mean(torch.square(di_L2_dt_loss))),
                        safe_op(torch.mean(torch.square(dv_c_dt_loss))),
                        safe_op(torch.mean(torch.square(v_out - v_out_attacked))),
                        safe_op(torch.mean(torch.square(v_out_constraint))),
                        safe_op(torch.mean(torch.square(di_out_dt - (1/L_dc) * (v_out - v_battery - R_battery * i_out)))),                      
                        safe_op(torch.mean(torch.square(P_dc - P_ac) + torch.square(P_out - P_dc * DC_DC_EFFICIENCY))),
                        safe_op(torch.mean(torch.square(i_ac - i_L2) + torch.square(i_d - i_d_measured) + torch.square(i_q - i_q_measured)))
                    ]
                    
                    evcs_loss.extend(evcs_losses)

                    # evcs_loss.backward(retain_graph=True)  

                    # # Clear gradients for next iteration
                    # delta.grad = None
                    # omega.grad = None
                    # phi_d.grad = None
                    # phi_q.grad = None
                    # i_d.grad = None
                    # i_q.grad = None
                    # i_L1.grad = None
                    # i_L2.grad = None
                    # v_c.grad = None
                    # v_dc.grad = None
                    # i_out.grad = None

                except Exception as e:
                    print(f"Error processing EVCS {i}: {e}")
                    continue

            # Calculate final losses
            V_regulation_loss = safe_op(voltage_regulation_loss)
            if len(evcs_loss) > 0:
                # Filter out None values and convert to tensors if needed
                valid_losses = []
                for loss in evcs_loss:
                    if loss is not None:
                        if not isinstance(loss, torch.Tensor):
                            loss = torch.tensor(loss, dtype=torch.float32)
                        valid_losses.append(loss)
                
                if len(valid_losses) > 0:
                    evcs_total_loss = safe_op(torch.sum(torch.stack(valid_losses)))
                else:
                    evcs_total_loss = torch.tensor(0.0, dtype=torch.float32)
            else:
                evcs_total_loss = torch.tensor(0.0, dtype=torch.float32)
            wac_loss = safe_op(torch.mean(torch.square(wac_error_vdc) + torch.square(wac_error_vout)))
            
            # Calculate final total loss only after all components are computed
            total_loss = safe_op(power_flow_loss + evcs_total_loss + wac_loss + V_regulation_loss)
            
            return (
                total_loss,
                power_flow_loss,
                evcs_total_loss,
                wac_loss,
                V_regulation_loss
            )
            
    except Exception as e:
        print(f"Error in physics loss: {e}")
        return (
            torch.tensor(float('inf')), 
            torch.tensor(0.0), 
            torch.tensor(0.0), 
            torch.tensor(0.0), 
            torch.tensor(0.0)
        )


def calculate_gradient(x, spacing):
    """Calculate gradient using finite differences with proper handling of small tensors."""
    try:
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            



        # If tensor is too small, pad it by repeating the last value
        if x.shape[0] < 2:
            x = torch.cat([x, x], dim=0)
            
        # Calculate gradient using forward difference
        # (x[1:] - x[:-1]) / spacing for each batch
        grad = (x[1:] - x[:-1]) / spacing
        
        # If original input was single value, return single value gradient
        if grad.shape[0] == 0:
            return torch.zeros_like(x[0])
            
        return grad
        
    except Exception as e:
        print(f"Error in calculate_gradient: {e}")
        return torch.zeros_like(x[0] if len(x) > 0 else x)

def train_step(model, optimizer, bus_data_batch, Y_bus_tf, bus_data_tf, attack_actions, defend_actions):
    """Performs a single training step with proper tensor handling"""
    try:
        # Zero gradients
        optimizer.zero_grad()
        
        # Calculate all losses
        total_loss, power_flow_loss, evcs_loss, wac_loss, v_reg_loss = physics_loss(
            model, Y_bus_tf, bus_data_tf,
            attack_actions, defend_actions
        )
        
        # Skip gradient update if we got error values
        if torch.abs(total_loss) >= 1e6:  # Check for error condition
            print("Skipping gradient update due to error in physics_loss")
            return torch.tensor(1e6, dtype=torch.float32)
        
        # Backward pass
        total_loss.backward()
        
        # Apply gradients
        optimizer.step()
        
        return total_loss, {
            'power_flow_loss': power_flow_loss,
            'evcs_loss': evcs_loss,
            'wac_loss': wac_loss,
            'v_reg_loss': v_reg_loss
        }
    except Exception as e:
        print(f"Error in training step: {e}")
        return torch.tensor(1e6, dtype=torch.float32), {}

def train_model(initial_model, dqn_agent, sac_attacker, sac_defender, Y_bus_tf, bus_data, epochs=1500, batch_size=256):
    """Train the PINN model with proper data handling."""
    try:
        model = initial_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create environment with necessary data
        env = CompetingHybridEnv(
            pinn_model=model,
            y_bus_tf=Y_bus_tf,
            bus_data=bus_data,
            v_base_lv=V_BASE_DC,
            dqn_agent=dqn_agent,
            num_evcs=NUM_EVCS,
            num_buses=NUM_BUSES,
            time_step=TIME_STEP
        )
        
        # Convert bus data to PyTorch tensors
        bus_data_tf = torch.tensor(bus_data, dtype=torch.float32)
        Y_bus_tf = torch.tensor(Y_bus_tf, dtype=torch.float32)    
        
        history = {
            'total_loss': [],
            'power_flow_loss': [],
            'evcs_loss': [],
            'wac_loss': [],
            'v_reg_loss': []
        }
        
        for epoch in range(epochs):
            try:
                # Reset environment and get initial state
                reset_result = env.reset()
                
                # Handle different return formats from reset()
                if isinstance(reset_result, tuple):
                    state = reset_result[0]
                else:
                    state = reset_result
                    
                if state is None:
                    print(f"Error: Invalid state in epoch {epoch}")
                    continue
                
                # Ensure state is properly shaped
                state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
                
                try:
                    # Get actions from agents
                    # DQN action
                    dqn_prediction = dqn_agent.predict(state.numpy(), deterministic=True)
                    dqn_action = dqn_prediction[0] if isinstance(dqn_prediction, tuple) else dqn_prediction
                    
                    # SAC Attacker action
                    attack_prediction = sac_attacker.predict(state.numpy(), deterministic=True)
                    attack_action = attack_prediction[0] if isinstance(attack_prediction, tuple) else attack_prediction
                    
                    # SAC Defender action
                    defend_prediction = sac_defender.predict(state.numpy(), deterministic=True)
                    defend_action = defend_prediction[0] if isinstance(defend_prediction, tuple) else defend_prediction
                    
                    # Convert actions to tensors
                    attack_tensor = torch.tensor(attack_action, dtype=torch.float32).reshape(1, -1)
                    defend_tensor = torch.tensor(defend_action, dtype=torch.float32).reshape(1, -1)
                    
                except Exception as e:
                    print(f"Error in action prediction: {str(e)}")
                    continue
                
                # Calculate losses
                try:
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass and loss calculation
                    losses = physics_loss(
                        model=model,
                        t=torch.tensor([[epoch * TIME_STEP]], dtype=torch.float32),
                        Y_bus_tf=Y_bus_tf,
                        bus_data=bus_data_tf,
                        attack_actions=attack_tensor,
                        defend_actions=defend_tensor
                    )
                    
                    if not isinstance(losses, tuple) or len(losses) != 5:
                        print(f"Invalid losses returned in epoch {epoch}")
                        continue
                        
                    total_loss, pf_loss, ev_loss, wac_loss, v_loss = losses
                    
                    # Update history
                    history['total_loss'].append(float(total_loss.item()))
                    history['power_flow_loss'].append(float(pf_loss.item()))
                    history['evcs_loss'].append(float(ev_loss.item()))
                    history['wac_loss'].append(float(wac_loss.item()))
                    history['v_reg_loss'].append(float(v_loss.item()))
                    
                    # Backward pass and optimization
                    if torch.isfinite(total_loss):
                        total_loss.backward()
                        optimizer.step()
                    
                except Exception as e:
                    print(f"\nDetailed Error Information for epoch {epoch}:")
                    print(f"Error Type: {type(e).__name__}")
                    print(f"Error Message: {str(e)}")
                    traceback.print_exc()
                    continue
                
                # Take environment step
                try:
                    next_state, rewards, done, truncated, info = env.step({
                        'dqn': dqn_action,
                        'attacker': attack_action,
                        'defender': defend_action
                    })
                    
                except Exception as e:
                    print(f"Error in environment step for epoch {epoch}: {str(e)}")
                    continue
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue
        
        return model, history
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return initial_model, None


def evaluate_model_with_three_agents(env, dqn_agent, sac_attacker, sac_defender, num_steps=1500):
    """Evaluate the environment with DQN, SAC attacker, and SAC defender agents."""
    try:
        state, _ = env.reset()
        done = False
        time_step = env.TIME_STEP if hasattr(env, 'TIME_STEP') else TIME_STEP  # Add fallback

        # Initialize tracking variables as lists
        tracking_data = {
            'time_steps': [],
            'cumulative_deviations': [],
            'voltage_deviations': [],
            'attack_active_states': [],
            'target_evcs_history': [],
            'attack_durations': [],
            'dqn_actions': [],
            'sac_attacker_actions': [],
            'sac_defender_actions': [],
            'observations': [],
            'evcs_attack_durations': {i: [] for i in range(env.NUM_EVCS)},
            'attack_counts': {i: 0 for i in range(env.NUM_EVCS)},
            'total_durations': {i: 0 for i in range(env.NUM_EVCS)},
            'rewards': []
        }

        for step in range(num_steps):
            current_time = step * time_step
            
            try:
                # Convert state to numpy if it's a tensor
                if isinstance(state, torch.Tensor):
                    state_np = state.detach().numpy()
                else:
                    state_np = np.array(state)

                # Get DQN action
                dqn_raw = dqn_agent.predict(state_np, deterministic=True)
                dqn_action = dqn_raw[0] if isinstance(dqn_raw, tuple) else dqn_raw
                
                # Convert DQN action to proper format
                if isinstance(dqn_action, np.ndarray):
                    dqn_action = torch.from_numpy(dqn_action).long()
                elif not isinstance(dqn_action, torch.Tensor):
                    dqn_action = torch.tensor(dqn_action, dtype=torch.long)

                # Get SAC actions
                sac_attacker_action, _ = sac_attacker.predict(state_np, deterministic=True)
                sac_defender_action, _ = sac_defender.predict(state_np, deterministic=True)
                
                # Convert SAC actions to numpy arrays
                if isinstance(sac_attacker_action, torch.Tensor):
                    sac_attacker_action = sac_attacker_action.detach().numpy()
                if isinstance(sac_defender_action, torch.Tensor):
                    sac_defender_action = sac_defender_action.detach().numpy()

                # Combine actions
                action = {
                    'dqn': dqn_action,
                    'attacker': sac_attacker_action,
                    'defender': sac_defender_action
                }

                # Take step in environment
                next_state, rewards, done, truncated, info = env.step(action)
                
                # Handle rewards properly
                if isinstance(rewards, dict):
                    # Sum up all rewards from all agents
                    reward_value = sum(value for value in rewards.values() if isinstance(value, (int, float)))
                else:
                    reward_value = float(rewards) if isinstance(rewards, (int, float)) else 0.0

                # Update tracking data
                tracking_data['rewards'].append(reward_value)
                
                # Ensure next_state is numpy array
                if isinstance(next_state, torch.Tensor):
                    next_state = next_state.detach().numpy()

                # Store data with proper type conversion
                tracking_data['time_steps'].append(float(current_time))
                tracking_data['cumulative_deviations'].append(float(info.get('cumulative_deviation', 0)))
                tracking_data['voltage_deviations'].append(
                    np.array(info.get('voltage_deviations', [0] * env.NUM_EVCS), dtype=np.float32)
                )
                tracking_data['attack_active_states'].append(bool(info.get('attack_active', False)))
                tracking_data['target_evcs_history'].append(
                    np.array(info.get('target_evcs', [0] * env.NUM_EVCS), dtype=np.float32)
                )
                tracking_data['attack_durations'].append(float(info.get('attack_duration', 0)))
                tracking_data['dqn_actions'].append(dqn_action.cpu().numpy() if isinstance(dqn_action, torch.Tensor) else dqn_action)
                tracking_data['sac_attacker_actions'].append(sac_attacker_action.tolist())
                tracking_data['sac_defender_actions'].append(sac_defender_action.tolist())
                tracking_data['observations'].append(next_state.tolist())

                # Track EVCS-specific attack data
                target_evcs = np.array(info.get('target_evcs', [0] * env.NUM_EVCS))
                attack_duration = float(info.get('attack_duration', 0))
                for i in range(env.NUM_EVCS):
                    if target_evcs[i] == 1:
                        tracking_data['evcs_attack_durations'][i].append(attack_duration)
                        tracking_data['attack_counts'][i] += 1
                        tracking_data['total_durations'][i] += attack_duration
                
                state = next_state
                if done or truncated:
                    break

            except Exception as e:
                print(f"Error in evaluation step {step}: {str(e)}")
                continue

        # Calculate average attack durations
        avg_attack_durations = []
        for i in range(env.NUM_EVCS):
            if tracking_data['attack_counts'][i] > 0:
                avg_duration = tracking_data['total_durations'][i] / tracking_data['attack_counts'][i]
            else:
                avg_duration = 0
            avg_attack_durations.append(float(avg_duration))

        # Convert lists to numpy arrays
        processed_data = {}
        for key, value in tracking_data.items():
            try:
                if isinstance(value, dict):
                    processed_data[key] = value
                elif key in ['time_steps', 'cumulative_deviations', 'attack_durations']:
                    processed_data[key] = np.array(value, dtype=np.float32)
                elif key in ['voltage_deviations', 'sac_attacker_actions', 'sac_defender_actions']:
                    processed_data[key] = np.array(value, dtype=np.float32)
                elif key == 'attack_active_states':
                    processed_data[key] = np.array(value, dtype=bool)
                else:
                    processed_data[key] = value
            except Exception as e:
                print(f"Error processing {key}: {str(e)}")
                processed_data[key] = value

        processed_data['avg_attack_durations'] = np.array(avg_attack_durations, dtype=np.float32)
        return processed_data

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None

def check_constraints(state, info):
        """Helper function to check individual constraints."""
        violations = []
        
        # Extract relevant state components
        # Assuming state structure matches your environment's observation space
        voltage_indices = slice(0, NUM_BUSES)  # Adjust based on your state structure
        current_indices = slice(NUM_BUSES, 2*NUM_BUSES)  # Adjust as needed
        
        # Check voltage constraints (0.9 to 1.1 p.u.)
        voltages = state[voltage_indices]
        if torch.any(voltages < 0.8) or torch.any(voltages > 1.2):
            violations.append({
                'type': 'Voltage',
                'values': voltages,
                'limits': (0.8, 1.2),
                'violated_indices': torch.where((voltages < 0.8) | (voltages > 1.2))[0]
            })

        # Check current constraints (-1.0 to 1.0 p.u.)
        currents = state[current_indices]
        if torch.any(torch.abs(currents) > 1.0):
            violations.append({
                'type': 'Current',
                'values': currents,
                'limits': (-1.0, 1.0),
                'violated_indices': torch.where(torch.abs(currents) > 1.0)[0]
            })

        # Check power constraints if available in state
        if 'power_output' in info:
            power = info['power_output']
            if torch.any(torch.abs(power) > 1.0):
                violations.append({
                    'type': 'Power',
                    'values': power,
                    'limits': (-1.0, 1.0),
                    'violated_indices': torch.where(torch.abs(power) > 1.0)[0]
                })

        # Check SOC constraints if available
        if 'soc' in info:
            soc = info['soc']
            if torch.any((soc < 0.1) | (soc > 0.9)):
                violations.append({
                    'type': 'State of Charge',
                    'values': soc,
                    'limits': (0.1, 0.9),
                    'violated_indices': torch.where((soc < 0.1) | (soc > 0.9))[0]
                })

        return violations, info

def validate_physics_constraints(env, dqn_agent, sac_attacker, sac_defender, num_episodes=5):
    """Validate that the agents respect physics constraints with detailed reporting."""


    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            try:
                # Get actions from all agents
                dqn_action_scalar = dqn_agent.predict(state, deterministic=True)[0]
                dqn_action = env.decode_dqn_action(dqn_action_scalar)
                attacker_action = sac_attacker.predict(state, deterministic=True)[0]
                defender_action = sac_defender.predict(state, deterministic=True)[0]
                
                # Combine actions
                action = {
                    'dqn': dqn_action,
                    'attacker': attacker_action,
                    'defender': defender_action
                }
                
                # Take step in environment
                next_state, rewards, done, truncated, info = env.step(action)
                
                # Check for physics violations
                violations = check_constraints(next_state, info)
                
                if violations:
                    print(f"\nPhysics constraints violated in episode {episode}, step {step_count}:")
                    for violation in violations:
                        print(f"\nViolation Type: {violation['type']}")
                        print(f"Limits: {violation['limits']}")
                        # print(f"Violated at indices: {violation['violated_indices']}")
                        # print(f"Values at violated indices: {violation['values'][violation['violated_indices']]}")
                    return False
                
                state = next_state
                step_count += 1
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                return False
            
    print("All physics constraints validated successfully!")
    return True, info


def convert_to_serializable(obj):
    """Convert numpy arrays and tensors to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj



def plot_evaluation_results(results, save_dir="./figures"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data from results and properly scale time
    time_steps = torch.tensor(results['time_steps']) * 10  # Multiply by 10 to correct scaling        
    cumulative_deviations = torch.tensor(results['cumulative_deviations'])
    voltage_deviations = torch.tensor(results['voltage_deviations'])
    attack_active_states = torch.tensor(results['attack_active_states'])
    avg_attack_durations = torch.tensor(results['avg_attack_durations'])

    # Plot cumulative deviations over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, cumulative_deviations, label='Cumulative Deviations')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Deviations')
    plt.title('Cumulative Deviations Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/cumulative_deviations_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot total rewards over time
    plt.figure(figsize=(12, 6))
    # Convert rewards to total numerical values if they're dictionaries
    total_rewards = []
    for reward in results['rewards']:
        if isinstance(reward, dict):
            total_rewards.append(reward.get('attacker', 0) + reward.get('defender', 0))
        else:
            total_rewards.append(float(reward))
    
    plt.plot(time_steps, total_rewards, label='Total Rewards')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot voltage deviations for each EVCS over time
    plt.figure(figsize=(12, 6))
    for i in range(voltage_deviations.shape[1]):
        plt.plot(time_steps, voltage_deviations[:, i], label=f'EVCS {i+1} Voltage Deviation')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage Deviation (p.u.)')
    plt.title('Voltage Deviations Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/voltage_deviations_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot attack active states over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, attack_active_states, label='Attack Active State')
    plt.xlabel('Time (s)')
    plt.ylabel('Attack Active State')
    plt.title('Attack Active State Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/attack_states_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot average attack durations for each EVCS
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(avg_attack_durations)), avg_attack_durations, 
            tick_label=[f'EVCS {i+1}' for i in range(len(avg_attack_durations))])
    plt.xlabel('EVCS')
    plt.ylabel('Average Attack Duration (s)')
    plt.title('Average Attack Duration for Each EVCS')
    plt.grid(True)
    plt.savefig(f"{save_dir}/avg_attack_durations_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    # Define physics parameters
    print("Starting program execution...")
    start_time = time.time()

    physics_params = {
        'voltage_limits': (0.5, 1.5),
        'v_out_nominal': 1.0,
        'current_limits': (-0.1, 1.0),
        'i_rated': 1.0,
        'attack_magnitude': 0.04,
        'current_magnitude': 0.03,
        'wac_kp_limits': (0.0, 0.5),
        'wac_ki_limits': (0.0, 0.5),
        'control_saturation': 0.3,
        'power_limits': (0.5, 1.5),
        'power_ramp_rate': 0.1,
        'evcs_efficiency': 0.98,
        'soc_limits': (0.1, 0.9),
        'modulation_index_system': (modulation_index_system*0.5, modulation_index_system*1.5)
    }

    # Initialize the PINN model
    initial_pinn_model = EVCS_PowerSystem_PINN()


    optimizer = torch.optim.Adam(initial_pinn_model.parameters(), lr=1e-3)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{timestamp}"
    model_dir = f"./models/{timestamp}"
    for dir_path in [log_dir, model_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Create the Discrete Environment for DQN Agent
    print("Creating the DiscreteHybridEnv environment...")
    discrete_env = DiscreteHybridEnv(
        pinn_model=initial_pinn_model,
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP,
        **physics_params
    )

    # Initialize callbacks
    
    dqn_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/dqn_checkpoints/",
        name_prefix="dqn"
    )
    
    # Initialize the DQN Agent with improved parameters
    print("Initializing the DQN agent...")
    dqn_agent = DQN(
        'MlpPolicy',
        discrete_env,
        verbose=1,
        learning_rate=3e-3,
        buffer_size=10000,
        exploration_fraction=0.3,
        exploration_final_eps=0.2,
        train_freq=4,
        batch_size=32,
        gamma=0.99,
        device='cuda',
        tensorboard_log=f"{log_dir}/dqn/"
    )

    # Train DQN with monitoring
    print("Training DQN agent...")
    dqn_agent.learn(
        total_timesteps=2500,
        callback=dqn_checkpoint,
        progress_bar=True
    )
    dqn_agent.save(f"{model_dir}/dqn_final")

    # Create the CompetingHybridEnv
    print("Creating the CompetingHybridEnv environment...")
    combined_env = CompetingHybridEnv(
        pinn_model=initial_pinn_model,
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        dqn_agent=dqn_agent,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP,
        **physics_params
    )

    print("Creating SAC Wrapper environments...")
    sac_attacker_env = SACWrapper(
        env=combined_env,
        agent_type='attacker',
        dqn_agent=dqn_agent
    )
    # Initialize SAC Attacker
    print("Initializing SAC Attacker...")
    sac_attacker = SAC(
        'MlpPolicy',
        sac_attacker_env,
        verbose=1,
        learning_rate=5e-4,
        buffer_size=10000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        device='cuda',
        tensorboard_log=f"{log_dir}/sac_attacker/"
    )

    # Create defender wrapper environment with the trained attacker
    print("Creating SAC Defender environment...")
    sac_defender_env = SACWrapper(
        env=combined_env,
        agent_type='defender',
        dqn_agent=dqn_agent
    )

    # Initialize SAC Defender
    print("Initializing SAC Defender...")
    sac_defender = SAC(
        'MlpPolicy',
        sac_defender_env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        device='cuda',
        tensorboard_log=f"{log_dir}/sac_defender/"
    )

    # Update wrapper environments with both agents
    sac_attacker_env.sac_defender = sac_defender
    sac_defender_env.sac_attacker = sac_attacker

    # Create callbacks for monitoring
    sac_attacker_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/sac_attacker_checkpoints/",
        name_prefix="attacker"
    )
    
    sac_defender_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/sac_defender_checkpoints/",
        name_prefix="defender"
    )
# New Addition 
    print("Training the SAC Attacker agent...")
    sac_attacker.learn(
        total_timesteps=5000,   
        callback=sac_attacker_checkpoint,
        progress_bar=True
    )
    sac_attacker.save(f"{model_dir}/sac_attacker_final")

    print("Training the SAC Defender agent...")
    sac_defender.learn(
        total_timesteps=2500,
        callback=sac_defender_checkpoint,
        progress_bar=True
    )
    sac_defender.save(f"{model_dir}/sac_defender_final")

    num_iterations = 10

    
    # Joint training loop with validation
    print("Starting joint training...")
    for iteration in range(num_iterations):
        print(f"\nJoint training iteration {iteration + 1}/{num_iterations}")
        
        # Train agents with progress monitoring
        for agent, name, callback, env in [
            (dqn_agent, "DQN", dqn_checkpoint, discrete_env),
            (sac_attacker, "SAC Attacker", sac_attacker_checkpoint, sac_attacker_env),
            (sac_defender, "SAC Defender", sac_defender_checkpoint, sac_defender_env)
        ]:
            print(f"\nTraining {name}...")
            if name == "SAC Defender":
                total_timesteps=2500
            else:
                total_timesteps=5000
            agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            agent.save(f"{model_dir}/{name.lower()}_iter_{iteration + 1}")

            # Update environment references after each agent training
            combined_env.update_agents(dqn_agent, sac_attacker, sac_defender)
            sac_attacker_env.update_agents(sac_defender=sac_defender, dqn_agent=dqn_agent)
            sac_defender_env.update_agents(sac_attacker=sac_attacker, dqn_agent=dqn_agent)

    epochs = 2500
    print("Training the PINN model with the hybrid RL agents (DQN for target, SAC Attacker for FDI, and SAC Defender for stabilization)...")
    trained_pinn_model, training_history = train_model(
        initial_model=initial_pinn_model,
        dqn_agent=dqn_agent,
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        Y_bus_tf=Y_bus_tf,  # Your Y-bus matrix
        bus_data=bus_data,  # Your bus data
        epochs=epochs,
        batch_size=128
    )

    # Optionally plot training history
    if training_history is not None:
        for epoch in range(0, epochs, 100):  # Print every 100 epochs
            print(f"\nEpoch {epoch}:")
            print(f"Total Loss: {training_history['total_loss'][epoch]:.4f}")
            print(f"Power Flow Loss: {training_history['power_flow_loss'][epoch]:.4f}")
            print(f"EVCS Loss: {training_history['evcs_loss'][epoch]:.4f}")
            print(f"WAC Loss: {training_history['wac_loss'][epoch]:.4f}")
            print(f"Voltage Regulation Loss: {training_history['v_reg_loss'][epoch]:.4f}")
    

        # After training the PINN model, create a new environment using the trained model
    print("Creating a new CompetingHybridEnv environment with the trained PINN model...")
    trained_combined_env = CompetingHybridEnv(
        pinn_model=trained_pinn_model,  # Use the trained PINN model here
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        dqn_agent=dqn_agent,  # Use the trained agents
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP
    )

    # Save the trained model if needed
    try:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(trained_pinn_model.state_dict(), f'models/pinn_model_{current_time}.pth')
        print(f"\nModel saved successfully as: pinn_model_{current_time}.pth")
        
        # Save training history
        import json
        with open(f'models/training_history_{current_time}.json', 'w') as f:
            json.dump(training_history, f)
        print(f"Training history saved as: training_history_{current_time}.json")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")

        # Update the environment's agent references if necessary
    trained_combined_env.sac_attacker = sac_attacker
    trained_combined_env.sac_defender = sac_defender
    trained_combined_env.dqn_agent = dqn_agent

    # Update the main evaluation and saving code
    try:
        print("Running final evaluation...")
        # Change this line:
        # results = evaluate_model_with_three_agents(env, dqn_agent, sac_attacker, sac_defender)
        
        # To this:
        results = evaluate_model_with_three_agents(trained_combined_env, dqn_agent, sac_attacker, sac_defender)
        
        if results is not None:
            # Convert results to serializable format
            save_results = convert_to_serializable(results)
            
            # Save to file
            with open('evaluation_results.json', 'w') as f:
                json.dump(save_results, f, indent=4)
            print("Evaluation results saved successfully")
            
            # Prepare data for plotting
            plot_data = {
                'time_steps': save_results['time_steps'],
                'cumulative_deviations': save_results['cumulative_deviations'],
                'voltage_deviations': save_results['voltage_deviations'],
                'attack_active_states': save_results['attack_active_states'],
                'target_evcs_history': save_results['target_evcs_history'],
                'attack_durations': save_results['attack_durations'],
                'observations': save_results['observations'],
                'avg_attack_durations': save_results['avg_attack_durations'],
                'rewards': save_results['rewards']
            }
            
            # Plot the results
            plot_evaluation_results(plot_data)
        else:
            print("Evaluation failed to produce results")
            
    except Exception as e:
        print(f"Error in final evaluation: {e}")
        traceback.print_exc()


    print("\nTraining completed successfully!")

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Format the time nicely
    time_delta = timedelta(seconds=execution_time)
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    
    print("\nProgram Execution Summary:")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")

