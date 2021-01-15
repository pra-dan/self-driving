import numpy as np
from math import *
import matplotlib.pyplot as plt
from ex1_sampling import sample_normal_distribution

def sample_motion_model(x, u, alpha):
    """
    xt0: current pose of the robot (befor moving)
    ut: odometry reading
    alpha: noise params

    returns: xt1: new pose
    """
    xt0, yt0, theta0 = x
    del_rot_1, del_rot_2, del_trans = u
    a1, a2, a3, a4 = alpha

    # Making the odometry values more realistic by adding noise
    app_del_rot_1 = del_rot_1 + sample_normal_distribution(0,a1*abs(del_rot_1) + a2*abs(del_trans))
    app_del_trans = del_trans + sample_normal_distribution(0,a3*del_trans + a4*(abs(del_rot_1)+abs(del_rot_2)))
    a5 = a1; a6 = a2 # just in this problem; otherwise maybe different
    app_del_rot_2 = del_rot_2 + sample_normal_distribution(0,a5*abs(del_rot_2) + a6*del_trans)

    # New Pose
    xt1 = xt0 + app_del_trans*cos(theta0+app_del_rot_1)
    yt1 = yt0 + app_del_trans*sin(theta0+app_del_rot_1)
    theta1 = theta0 + app_del_rot_1 + app_del_rot_2

    return xt1, yt1, theta1

def main():
    x = (2.0, 4.0, 0.0)
    u = (pi/2, 0.0, 1.0)
    alpha = (0.1, 0.1, 0.01, 0.01)

    X = list()
    Y = list()
    plt.figure(figsize=(7,7))
    for i in range(5000):
        xt1, yt1, _ = sample_motion_model(x, u, alpha)
        X.append(xt1)
        Y.append(yt1)

    plt.scatter(x[0],x[1]) #initial position
    plt.scatter(X,Y)       #expected positions
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.show()

if __name__ == "__main__":
    main()
