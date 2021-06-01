#discrete_filter.py
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def discrete_filter(bel, direction):
    '''
    This function skips the actions taken for a perceptual data, as we are
    only provided with action data

    PSEUDOCODE:
        For all x do:
            Bel'(x) = Sum( P(x|u,x')*Bel(x') ) for all x'

    //x' (aka previous pose) are the i-1, i-2 and i position. (i=current position)
    '''
    # Create empty array
    bel_prime = np.zeros(bel.shape[0])

    if(direction==+1):
        for i in range(bel.shape[0]):
            # 2 cell back belief
            if(i>=2):
                bel2 = bel[i-2]
            else:
                bel2 = 0

            # 1 cell back belief
            if(i>=1):
                bel1 = bel[i-1]
            else:
                bel1 = 0

            # curr cell belief
            bel0 = bel[i]

            # Cases
            ## Case 1: Normal. Current position < last cell
            if(i < bel.shape[0]-1):
                bel_prime[i] = 0.25*bel2 + 0.5*bel1 + 0.25*bel0
            elif(i==bel.shape[0]-1):
                ## Case 2: Current position = last cell
                bel_prime[i] = 0.25*bel2 + 0.75*bel1 + 1.0*bel0

    if(direction==-1):
        for i in range(bel.shape[0]):
            # 2 cells forward belief
            if(i < bel.shape[0]-2):
                bel2 = bel[i+2]
            else:
                bel2 = 0

            # 1 cell ahead belief
            if(i < bel.shape[0]-1):
                bel1 = bel[i+1]
            else:
                bel1 = 0

            # curr cell belief
            bel0 = bel[i]

            # Cases
            if(i > 0):
                ## Case 1: Normal. Current position > first cell
                bel_prime[i] = 0.25*bel2 + 0.5*bel1 + 0.25*bel0
            elif(i==0):
                ## Case 2: Current position = first cell
                bel_prime[i] = 0.25*bel2 + 0.75*bel1 + 1.0*bel0

    return bel_prime

def plot_belief(bel):
    plt.cla()
    plt.bar(range(0,bel.shape[0]), bel, width=1.0)
    plt.axis([0,bel.shape[0]-1,0,1])
    plt.draw()
    plt.pause(1)


def main():
    # Intial belief
    bel = np.hstack((np.zeros(10), 1, np.zeros(9)))
    plt.ion()

    for i in range(0,9):
        plot_belief(bel)
        # Go forward
        bel = discrete_filter(bel,+1)
        # Perform Check. The sum should be = 1
        print(np.sum(bel))

    for i in range(0,3):
        plot_belief(bel)
        # Go reverse
        bel = discrete_filter(bel,-1)
        # Perform Check. The sum should be = 1
        print(np.sum(bel))

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
