from layers import *

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """  
    
    ###########################################################################
    # TODO: Implement the affine_relu forward pass                            #
    ###########################################################################

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    ###########################################################################
    # TODO: Implement the affine_relu backwar pass                            #
    ###########################################################################

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx, dw, db
