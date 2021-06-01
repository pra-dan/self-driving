import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data

#add random seed for generating comparable pseudo random numbers
np.random.seed(123)

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.

    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles

def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound
    # (jump from -pi to pi). Therefore, we generate unit vectors from the
    # angles and calculate the angle of their average

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def sample_normal_distribution(sigma):
    try:
        return 0.5*np.sum(np.random.uniform(low=-sigma,high=sigma, size=12))
    except Exception as e:
        print(f'In sample_normal_distribution | error: {e}')
        return 0

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise
    # (probabilistic motion models slide 27)

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
    new_particles = []

    for p in particles:
        new_particle = dict()

        # Approximate using odometry + noise
        delta_rot1_cap = delta_rot1 + sample_normal_distribution(alpha1*abs(delta_rot1) + alpha2*delta_trans)
        delta_trans_cap = delta_trans + sample_normal_distribution(alpha3*delta_trans + alpha4*(abs(delta_rot1)+abs(delta_rot2)))
        delta_rot2_cap = delta_rot2 + sample_normal_distribution(alpha1*abs(delta_rot2) + alpha2*delta_trans)

        new_particle['x'] = p['x'] + delta_trans_cap*np.cos(p['theta']+delta_rot1_cap) #x_hat = old_x + r*Cos(theta)
        new_particle['y'] = p['y'] + delta_trans_cap*np.sin(p['theta']+delta_rot1_cap) #y_hat = old_y + r*Sin(theta)
        new_particle['theta'] = p['theta'] + delta_rot1_cap + delta_rot2_cap #theta_hat

        new_particles.append(new_particle)

    return new_particles

def eval_sensor_model(sensor_data, particles, landmarks):
    '''
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # Refer to probabilistic sensor models slide 33 or `landmark_detection_model(z,x,m)`
    done on 15/3/21 in notes. The arguments are:
        # z: sensor_data is generally given as z=<id,distance,bearing>
        The employed sensor model is range only... so no bearing !!! WTF
        # x: Each particle contains the info: x, y and theta
        # m: map or landmarks here. Each landmak contains lx & ly a.k.a mx & my in notes
    '''

    sigma_r = 0.2

    #measured landmark ids and ranges (aka distances)
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []

    for p in particles:
        pdet_pdt = 1.0 # NOTE: Product of all pdet
        # Iterate over all landmarks via ids
        for i,id in enumerate(ids):
            # coordinates of landmark
            mx, my = landmarks[id][0], landmarks[id][1]
            # coordinate of particle
            px, py = p['x'], p['y']

            # Find expected measurements
            dist_cap = np.sqrt(np.square(mx-px) + np.square(my-py))
            #bearing_cap =  not provided ...

            # Calculate Pdet or sensor model
            measurement_range = ranges[i]
            likelihood = scipy.stats.norm.pdf(measurement_range, dist_cap, sigma_r)
            # format: Prob(measured_dist, calculated_dist, sigma)

            # Same for bearing..
            # Not provided...

            # Combine individual measurements
            pdet_pdt = pdet_pdt * likelihood

        # Add to particle weight
        weights.append(pdet_pdt)

    # Normalize weights
    print(f'weights: {weights}\n\n')
    normalizer = sum(weights)
    weights = weights/normalizer

    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    # Initialize cdf or simply c
    c = weights[0] # Signifies where we are along the weights

    # Assign step size for u
    step = 1.0/len(particles)

    # Randomly choose starting pointer aka threshold
    u = np.random.uniform(0,1.0/len(particles))

    i = 0
    for p in particles:
        while(u > c):
            i = i+1
            c += weights[i]

        # Out of the while loop; if (cumulative weights or c > threshold)
        new_particles.append(particles[i]) # Note: p is not chosen here
        u += step

    return new_particles

def main():
    # implementation of a particle filter for robot pose estimation

    # Read landmarks
    ## The returned dict contains a list of landmarks each with the
    ## following information: {id, [x, y]}
    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    # Read sensor data
    ## The data contains {odometry,sensor} data accessed as:
    ## odometry_data = sensor_readings[timestep, 'odometry']
    ## sensor_data = sensor_readings[timestep, 'sensor']
    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    #initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    #run particle filter
    for timestep in range(int(len(sensor_readings)/2)):

        #plot the current state
        plot_state(particles, landmarks, map_limits)

        #predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep,'odometry'], particles)
        ##print(f'new_particles: \n{new_particles}')

        #calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)
        ##print(f'weights: {weights}')

        #resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show()

if __name__ == "__main__":
    main()
