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
	# standard deviations of motion noise
	sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
	sigma_delta_trans = noise[2] * delta_trans + \
		noise[3] * (abs(delta_rot1) + abs(delta_rot2))
	sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans
	# "move" each particle according to the odometry measurements plus sampled noise
	# to generate new particle set
	new_particles = []

	for particle in particles:
		new_particle = dict()
		#sample noisy motions
		noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
		noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
		noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)
		#calculate new particle pose
		new_particle['x'] = particle['x'] + \
			noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
		new_particle['y'] = particle['y'] + \
			noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
		new_particle['theta'] = particle['theta'] + \
			noisy_delta_rot1 + noisy_delta_rot2
		new_particles.append(new_particle)

		return new_particles

def eval_sensor_model(sensor_data, particles, landmarks):
	# Computes the observation likelihood of all particles, given the
	# particle and landmark positions and sensor measurements
	# (probabilistic sensor models slide 33)
	#
	# The employed sensor model is range only.
	sigma_r = 0.2
	#measured landmark ids and ranges
	ids = sensor_data['id']
	ranges = sensor_data['range']
	weights = []
	#rate each particle
	for particle in particles:
		all_meas_likelihood = 1.0 #for combini
		#loop for each observed landmark
		for i in range(len(ids)):
			lm_id = ids[i]
			meas_range = ranges[i]
			lx = landmarks[lm_id][0]
			ly = landmarks[lm_id][1]
			px = particle['x']
			py = particle['y']
			#calculate expected range measurement
			meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
			#evaluate sensor model (probability density function of normal distribution)
			meas_likelihood = scipy.stats.norm.pdf(meas_range, meas_range_exp, sigma_r)
			#combine (independent) measurements
			all_meas_likelihood = all_meas_likelihood * meas_likelihood
		weights.append(all_meas_likelihood)

	print(f'weights: {weights}\n\n')
    normalizer = sum(weights)
    weights = weights/normalizer

    return weights

def resample_particles(particles, weights):
	# Returns a new set of particles obtained by performing
	# stochastic universal sampling, according to the particle
	# weights.
	new_particles = []
	# distance between pointers
	step = 1.0/len(particles)
	# random start of first pointer
	u = np.random.uniform(0,step)
	# where we are along the weights
	c = weights[0
	# index of weight container and corresponding particle
	i = 0
	new_particles = []
	#loop over all particle weights
	for particle in particles:
		#go through the weights until you find the particle
		#to which the pointer points
		while u > c:
			i = i + 1
			c = c + weights[i]
		#add that particle
		new_particles.append(particles[i])
		#increase the threshold
		u = u + step

	return new_particles

def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

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
