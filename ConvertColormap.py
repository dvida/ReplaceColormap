""" Loads an image with one colour map and dumps it with a new colour map. """

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import scipy.optimize



def colorToValResiduals(params, cmap, target_rgb):
    """ Given a value between 0 and 1, a colormap, and the target RGB tuple, compute the residuals between 
        them. """

    # Get the predicted color
    rgb = cmap(params[0])

    # Compute the cost for every colour
    cost = (rgb[0] - target_rgb[0])**2 + (rgb[1] - target_rgb[1])**2 + (rgb[2] - target_rgb[2])**2

    return cost



def replaceCmap(rbg_in, cmap_in, cmap_out):
    """ Given the RGB value in the input color map, it find the corresponding value in the output colormap. 
    """

    # Don't replace grayscale
    if rbg_in[0] == rbg_in[1] == rbg_in[2]:
        return rbg_in


    ### Find which value maps to what colour in the old colour map by minimization ###

    # Find the initial minimization starting point by doing a grid search
    solutions = {}
    n_values = 100
    value_list = np.linspace(0.0, 1.0, n_values)
    for p0 in value_list:

        fun = colorToValResiduals([p0], cmap_in, rbg_in)
        solutions[fun] = p0

        

    # Choose the best solution
    val = solutions[min([key for key in solutions])]

    # Refine value by minimization
    res = scipy.optimize.basinhopping(colorToValResiduals, [val], niter=20, stepsize=0.1/n_values, \
        minimizer_kwargs={'args': (cmap_in, rbg_in), 'bounds': [[0.0, 1.0]], 'method': 'SLSQP'})
    val = res.x[0]


    ###

    
    # Get RGB using the new color map
    rgb_out = cmap_out(val)[:3]

    return rgb_out




if __name__ == "__main__":


    # Load the input image
    file_path = "test.png"

    # Input colormap
    input_cmap = 'jet'

    # Output color map
    out_cmap = 'viridis'



    # Load the image
    img_input = matplotlib.image.imread(file_path)

    # Init the colour maps
    cmap_in = matplotlib.cm.get_cmap(input_cmap)
    cmap_out = matplotlib.cm.get_cmap(out_cmap)


    # Init output image
    img_out = np.zeros_like(img_input)


    # Go through all image pixels and compute values of the new color map
    for i in range(img_input.shape[0]):
        print("Row: {:d}/{:d}".format(i, img_input.shape[0]))
        for j in range(img_input.shape[1]):
            
            new_rgb = replaceCmap(img_input[i, j], cmap_in, cmap_out)

            # print('----')
            # print("{:>4d}, {:>4d}".format(i, j))
            # print('In:', img_input[i, j])
            # print('Out:', new_rgb)

            img_out[i, j] = new_rgb



    plt.imshow(img_out)
    plt.show()

    # Save the image with the replaced color map
    matplotlib.image.imsave(file_path.split('.')[0] + '_{:s}.png'.format(out_cmap), img_out)