import matplotlib.pyplot as plt
import numpy as np

meas = [101, 82, 91, 112, 99, 151, 96, 85, 99, 105]

def inverse_sensor_model(measurement, cell):
    if(cell > measurement-10): # 10 = grid_resolution/2. IMO just bcoz measurement
    # is taken from the center of the cell
        return np.log(0.6/(1.-0.6))
    return np.log(0.3/(1.-0.3))

def occupancy_grid(c,m):
    global meas

    # Compute prior
    # Initial prior of the cells, the non informative prior of p = 0.5 will
    # yield 0 in log-odds form, so technically we could leave it out in the
    # the log-odds update formula
    prior = np.log(0.5 / (1 - 0.5))
    print(f'prior: {prior}')

    # Integrate each measurement
    for i in range(len(meas)):
        # Update each cell
        for j in range(len(c)):
            # If mi in perceptual field of z i.e.,
            # Anything beyond 20cm of the measurement should not be updated
            if(c[j] <= meas[i] + 20):
                m[j] = m[j] + inverse_sensor_model(meas[i], c[j]) - prior

    # Convert logOdds (m) to simple form
    m = 1 - 1./(1+np.exp(m))

    return m

def main():
    # Prepare 1D grid
    c = range(0,201,10)

    # Declare belief array
    m = np.zeros(len(c))

    # Update belief according to measurements
    m = occupancy_grid(c,m)

    plt.plot(c,m)
    plt.show()

if __name__ == '__main__':
    main()
