import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import subprocess
import time
from typing import Optional
import math

class X3UavRl(gym.Env):
    def __init__(self):
        super().__init__()
        subprocess.run("cd /<path_to_spawn.sh_file> && ./spawn.sh", shell=True)

        rclpy.init(args=None)
        self.node = Node("X3_uav_gazebo_rl_env")

        self.obs = self.node.create_subscription(Odometry,'/model/x3/odometry',self.Odometry_callback,10)
        self.pub = self.node.create_publisher(Twist,'/X3/gazebo/command/twist',10)

        self.quat = np.array([0,0,0,1])
        self.pos = np.array([0,0,0])

        self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(6,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10.0,high=10.0,shape=(7,),dtype=np.float32)

        self.pos_error = 0.0
        self.ori_error = 0.0
        self.reward = 0.0
        self.max_time = 30
        self.current_step = 0

        self.MAX_vel = 1.0
        self.MAX_ang_vel = 3.0
        self.CONTROL_DT = 0.05

    def Odometry_callback(self,msg:Odometry):
        self.pos[0] = float(msg.pose.pose.position.x)
        self.pos[1] = float(msg.pose.pose.position.y)
        self.pos[2] = float(msg.pose.pose.position.z)

        self.quat[0] = float(msg.pose.pose.orientation.x)
        self.quat[1] = float(msg.pose.pose.orientation.y)
        self.quat[2] = float(msg.pose.pose.orientation.z)
        self.quat[3] = float(msg.pose.pose.orientation.z)

        self._last_odom_time = time.time()

    def _get_observation(self):
        return np.concatenate([self.quat,self.pos]).astype(np.float32)
    
    def _get_info(self):
        self.ori_error = np.linalg.norm(self.quat - np.array([0.0,0.0,0.0,1.0],dtype=np.float32))
        self.pos_error = np.linalg.norm(self.pos - np.array([0.0,0.0,2.0],dtype=np.float32))
        return {"pos_error":self.pos_error, "ori_error":self.ori_error}

    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        cmd = [
            "gz", "service",
            "-s", "world/quadcopter/control",
            "--reqtype", "gz.msgs.WorldControl",
            "--reptype", "gz.msgs.Boolean",
            "--req", "reset: {all: true}"
        ]
        subprocess.run(cmd)

        start = time.time()
        while time.time() - start<2.0:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if getattr(self,"_last_odom_time",0)>start:
                break
        
        self.current_step = 0
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        x_vel = float(np.clip(action[0],-1,1))*self.MAX_vel
        y_vel = float(np.clip(action[1],-1,1))*self.MAX_vel
        z_vel = float(np.clip(action[2],-1,1))*self.MAX_vel
        x_ang_vel = float(np.clip(action[3],-1,1))*self.MAX_ang_vel
        y_ang_vel = float(np.clip(action[4],-1,1))*self.MAX_ang_vel
        z_ang_vel = float(np.clip(action[5],-1,1))*self.MAX_ang_vel

        msg = Twist()
        msg.linear.x = x_vel
        msg.linear.y = y_vel
        msg.linear.z = z_vel
        msg.angular.x = x_ang_vel
        msg.angular.y = y_ang_vel
        msg.angular.z = z_ang_vel
        self.pub.publish(msg)

        t0 = time.time()
        while time.time() - t0 < self.CONTROL_DT:
            rclpy.spin_once(self.node,timeout_sec=0.001)
        
        self.current_step+=1
        observtion = self._get_observation()
        info = self._get_info()

        pos_error = info["pos_error"]
        ori_error = info["ori_error"]
        reward = -5.0* pos_error - 0.5* ori_error + 1.5

        terminated = False
        truncated = False

        if pos_error < 0.2:
            reward+= 200

        if abs(observtion[4])>5.0 or abs(observtion[5])>5.0 or abs(observtion[6])>8.0:
            reward = -1000
            terminated = True

        if self.current_step >= int(self.max_time / self.CONTROL_DT):
            truncated = True
        
        return observtion , float(reward), terminated, truncated, info
    
from gymnasium.envs.registration import register
register(id='X3_uav-V0', entry_point='env:X3UavRl')
