import numpy as np
from numpy import linalg as LA
from physics_sim import PhysicsSim

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1 # action repeat not needed since velocities part of state

        self.state_size = self.action_repeat * 12 # was: 6
        self.action_low =  0 # 1 # 1
        self.action_high = 1 # 900 # 900
        self.action_low_all = 1
        self.action_high_all = 900
        self.action_size = 5  # 4 rotors and one average

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self.target_angle = np.array([0.,0.,0.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # minimize distance to target
        # reward for velocity towards target needed?
        # reward for low angle velocities
        vector_to_target = self.target_pos - self.sim.pose[:3] 
        distance_to_target = ((vector_to_target **2).sum())**0.5
        angular_v_measure = (self.sim.angular_v**2).sum() # ((self.sim.angular_v**2).sum())**0.5
        vel_vector_to_target_inner = np.inner(self.sim.v,vector_to_target)
        vel_vector_to_target_inner_normalized = 0. if abs(vel_vector_to_target_inner)<0.001 else \
            vel_vector_to_target_inner/(LA.norm(self.sim.v)*LA.norm(vector_to_target))
            
        angle_diff = self.sim.pose[3:]-self.target_angle
        angle_diff_plus_2pi = angle_diff + 2 *np.pi
        angle_diff_min_2pi = angle_diff - 2 *np.pi
        angle_diff_min = np.minimum(np.minimum(abs(angle_diff), abs(angle_diff_plus_2pi)), abs(angle_diff_min_2pi))
        angle_measure = angle_diff_min.sum()
         
        rd = 0.001 
        rav= 0.001
        rv = 0.001
        ra= 0.001
        cn = 0.1
        factor = 0.001
        
        #print("vel_vector_to_target_inner = {:6.2f}, vel_vector_to_target_inner_normalized = {:6.2f}".format(vel_vector_to_target_inner, vel_vector_to_target_inner_normalized))
        #reward = - rd * distance_to_target - ra*angular_v_measure
        #reward =  - rd * distance_to_target + rv * vel_vector_to_target_inner_normalized  - rav*angular_v_measure
        #reward =  rv * vel_vector_to_target_inner_normalized # - rav*angular_v_measure
        # reward = rd/(distance_to_target + 0.01)# + rav/(angular_v_measure+0.01)
        #reward = -ra * angle_measure
        #reward =  - rd * distance_to_target  + rv*vel_vector_to_target_inner_normalized  -rav * angular_v_measure - ra*angle_measure
        #reward = -rav * angular_v_measure
        reward = -rd * distance_to_target + rv*vel_vector_to_target_inner_normalized -ra*angle_measure -rav*angular_v_measure
        if distance_to_target < 0.5:
            reward += 0.0
        #reward = np.tanh(reward) # to get a reward between -1 and 1
        return factor*reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""

        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            if done:
                reward += -0.0
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.v) # added
            pose_all.append(self.sim.angular_v) # added
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        # TODO: extend state by v
        self.sim.reset()
        state = np.concatenate(([self.sim.pose]+[self.sim.v]+[self.sim.angular_v]) * self.action_repeat) # added v, angular_v
        return state