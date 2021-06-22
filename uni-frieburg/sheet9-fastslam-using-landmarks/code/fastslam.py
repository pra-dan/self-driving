from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def initialize_particles(num_particles, num_landmarks):
    #initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        #initial weight
        particle['weight'] = 1.0 / num_particles

        #particle history aka all visited poses
        particle['history'] = []

        #initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #initialize the landmark mean and covariance
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #add landmarks to particle
        particle['landmarks'] = landmarks

        #add particle to set
        particles.append(particle)

    return particles

def sample_normal_distribution(sigma):
    try:
        return 0.5*np.sum(np.random.uniform(low=-sigma,high=sigma, size=12))
    except Exception as e:
        print(f'In sample_normal_distribution | error: {e}')
        return 0

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]
    alpha1 = noise[0]
    alpha2 = noise[1]
    alpha3 = noise[2]
    alpha4 = noise[3]

    # generate new particle set after motion update
    for p in particles:

        # Approximate pose using odometry + noise
        delta_rot1_cap = delta_rot1 + sample_normal_distribution(alpha1*abs(delta_rot1) + alpha2*delta_trans)
        delta_trans_cap = delta_trans + sample_normal_distribution(alpha3*delta_trans + alpha4*(abs(delta_rot1)+abs(delta_rot2)))
        delta_rot2_cap = delta_rot2 + sample_normal_distribution(alpha1*abs(delta_rot2) + alpha2*delta_trans)

        p['x'] += delta_trans_cap*np.cos(p['theta']+delta_rot1_cap) #x_hat = old_x + r*Cos(theta)
        p['y'] += delta_trans_cap*np.sin(p['theta']+delta_rot1_cap) #y_hat = old_y + r*Sin(theta)
        p['theta'] += delta_rot1_cap + delta_rot2_cap #theta_hat

    return particles


def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.
    #For reference, see line 132 of `correction_step` of kalman_filter.py

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h
    # wrt the landmark location

    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H

def angle_diff(meas_bearing,exp_bearing):
    print(f'meas_bearing: {meas_bearing}\texp_bearing: {exp_bearing}')

    return (meas_bearing-exp_bearing) # normalize it

def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    Q_t = np.array([[0.1, 0],\
                    [0, 0.1]])

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta']

        #loop over observed landmarks
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]

            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                #initialize the landmark mean and covariance
                lx = px + meas_range*np.cos(meas_bearing+ptheta)
                ly = py + meas_range*np.sin(meas_bearing+ptheta)

                landmark['mu'][0] = lx
                landmark['mu'][1] = ly

                h, H = measurement_model(particle, landmark)

                #Compute covariance
                H_inv = np.linalg.inv(H)
                landmark['sigma'] = np.dot(np.dot(H_inv, Q_t), H_inv.T)

                #Mark landmark as observed
                landmark['observed'] = True ##

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above.
                # calculate particle weight: particle['weight'] = ...

                # EKF-update
                h, H = measurement_model(particle, landmark) # h aka expected measurement
                sigma = landmark['sigma']

                # Calculate measurement covariance and Kalman gain
                Q_t = np.dot(np.dot(sigma, H),  H.T) + Q_t
                Q_inv = np.linalg.inv(Q_t)
                K = np.dot(np.dot(sigma, H.T), Q_inv)

                # Update estimated landmark mean and covariance
                # Compute the difference between the observed and the expected measurement
                delta = np.array([meas_range - h[0], angle_diff(meas_bearing,h[1])])
                landmark['mu'] += K.dot(delta)
                landmark['sigma'] =  (np.identity(2) - K.dot(H)).dot(sigma)

                # Calculate weights
                '''
                det = np.linalg.det(2 * np.pi * Q_t)
                expo = np.exp(-0.5*(delta.T).dot(Q_inv).dot(delta))
                particle['weight'] *= 1/np.sqrt(det)*expo
                '''
                fact = 1 / np.sqrt(math.pow(2*math.pi,2) * np.linalg.det(Q_t))
                expo = -0.5 * np.dot(delta.T, np.linalg.inv(Q_t)).dot(delta)
                weight = fact * np.exp(expo)

                particle['weight'] *= weight


    #normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer

    return

def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle
    # weights.

    new_particles = []

    # Initialize cdf or simply c
    c = particles[0]['weight'] # Signifies where we are along the weights

    # Assign step size for u
    step = 1.0/len(particles)

    # Randomly choose starting pointer aka threshold
    u = np.random.uniform(0,step)

    i = 0
    for p in particles:
        #go through the weights until you find the particle
        #to which the pointer points
        while(u > c):
            i = i+1
            c += particles[i]['weight']

        # Out of the while loop; if (cumulative weights or c > threshold)

        # hint: To copy a particle from particles to the new_particles
        # list, first make a copy:
        # new_particle = copy.deepcopy(particles[i])
        new_particle = copy.deepcopy(particles[i])
        new_particle['weight'] = 1/len(particles)
        new_particles.append(new_particle) # Note: p is not chosen here
        u += step

    return new_particles

def main():

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    #create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    #run FastSLAM
    for timestep in range(int(len(sensor_readings)/2)):

        #predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        #plot filter state
        print(f'\nparticles: {particles}\n')
        plot_state(particles, landmarks)

        #calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show('hold')

if __name__ == "__main__":
    main()
