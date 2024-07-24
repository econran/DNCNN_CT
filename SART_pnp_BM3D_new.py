#——————————————————————————————————————————————————————————————————————————————#
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp                 #
#            2013-2018, CWI, Amsterdam                                         # 
#                                                                              # 
# Contact: astra@astra-toolbox.com                                             # 
# Website: http://www.astra-toolbox.com/                                       # 
#——————————————————————————————————————————————————————————————————————————————#

#——————————————————————————————————————————————————————————————————————————————#
# Plug-n-Play Superiorization code by Dr. Thomas Humphries, UW Bothell         # 
# BM3D modifications by by Dr. Thomas Humphries & Jonathan Henshaw             #   
# Fall 2021 Undergraduate Research                                             #
#——————————————————————————————————————————————————————————————————————————————#

#——————————————————————————————————————————————————————————————————————————————#
# Import                                                                       #
#——————————————————————————————————————————————————————————————————————————————#
#importing the libraries
import argparse
from glob import glob
import os
from PIL import Image
import astra
import numpy as np
from bm3d import bm3d

#——————————————————————————————————————————————————————————————————————————————#
# Function Definitions                                                         #
#——————————————————————————————————————————————————————————————————————————————#

# This function creates projectors, preprocessing our data before
# passing it to astra for the heavy lifting, accurate, letting astra
#Minor modifications, at most
def create_projector(geom, numbin, angles, dso, dod, fan_angle):
    if geom == 'parallel':
        proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
    elif geom == 'fanflat':
        #convert to mm for astra
        dso *=10; dod *=10;
        #compute tan of 1/2 the fan angle
        ft = np.tan( np.deg2rad(fan_angle / 2) )
        #width of one detector pixel, calculated based on fan angle
        det_width = 2 * (dso + dod) * ft / numbin

        proj_geom = astra.create_proj_geom\
                    (geom, det_width, numbin, angles, dso, dod)

    p = astra.create_projector('cuda',proj_geom,vol_geom); #make new variable, that is not p because p is used elsewhere
    return p

# This function builds and initializes the argument parser,
# then returns the parsed arguments as they were provided on
# the command line
def generateParsedArgs():
    #Initialize parser
    parser = argparse.ArgumentParser(description='') #Add the argument

    parser.add_argument('--sino', dest='infile', default='.', \ #Add the sinogram, sinos folder
        help='input sinogram -- directory or single file')

    parser.add_argument('--out', dest='outfolder', default='.', \ #Outputs folder
        help='output directory')

    parser.add_argument('--numpix', dest='numpix', type=int, default=512, \ #Default the number of pixels to 512
        help='size of volume (n x n )')

    parser.add_argument('--psize', dest='psize', default='', \ #Defining pixel size  
        help='pixel size (float) OR file containing pixel sizes (string)');

    parser.add_argument('--numbin', dest='numbin', type=int, default=729, \ #Number of detector pixels, number of columns in system matrix
        help='number of detector pixels')

    parser.add_argument('--ntheta', dest='numtheta', type=int, default=900, \
        help='number of angles')

    parser.add_argument('--nsubs', dest='ns', type=int, default=1, \ # number of subsets, cannot have decimal array
        help='number of subsets. must divide evenly into number of angles')

    parser.add_argument('--range', dest='theta_range', type=float, nargs=2, \ #Angle range
                        default=[0, 360], \
        help='starting and ending angles (deg)')

    parser.add_argument('--geom', dest='geom', default='fanflat', \ #not necessarily worrying about rn. 
        help='geometry (parallel or fanflat)')

    parser.add_argument('--dso', dest='dso', type=float, default=100, \ #projection geometry
        help='source-object distance (cm) (fanbeam only)')

    parser.add_argument('--dod', dest='dod', type=float, default=100, \
        help='detector-object distance (cm) (fanbeam only)')

    parser.add_argument('--fan_angle', dest='fan_angle', default=35, type=float, \
        help='fan angle (deg) (fanbeam only)')

    parser.add_argument('--numits', dest='num_its', default=32, type=int, \ #For the superiorization, default is 32
        help='maximum number of iterations')

    parser.add_argument('--beta', dest='beta', default=1., type=float, \ #Relaxation parameter
        help='relaxation parameter beta')

    parser.add_argument('--x0', dest='x0_file',default='', \ #initial image, default to 0 
        help='initial image (default: zeros)')

    parser.add_argument('--xtrue', dest='xtrue_file', default='', \ #The true image, trying to get to 
        help='true image (if available)')

    parser.add_argument('--sup_params', dest='sup_params', type=float, nargs=4,\ #kmin, kstep, gamma, and bm3D is not important to our code. 
        help='superiorization parameters: k_min, k_step, gamma, bm3d_sigma')

    parser.add_argument('--epsilon_target', dest='epsilon_target', default=0., \ #episilon target is just 0
        help='target residual value (float, or file with residual values)')

    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./checkpoint', \ #Unsure what this is.
        help='directory containing checkpoint for DnCnn')

    parser.add_argument('--make_png', dest='make_png',type=bool, default=False,\
        help='whether or not you would like to generate .png files')

    parser.add_argument('--make_intermediate', dest='make_intermediate', \ #Make steps along the way to show how well things are going 
                        type=bool, default=False,\
        help='whether or not you would like to generate output files each iter')

    parser.add_argument('--overwrite', dest='overwrite', \
                        type=bool, default=True,\
        help='whether you would like to reprocess preexisting files on export')

    #Return arguments as parsed from command line
    return parser.parse_args()


# This function outputs a .png, we have stuff to make both .flt and .png
def makePNG(f, outname):
    #Set any negative values to positive machine epsilon
    img = np.maximum(f,np.finfo(float).eps)
    #Scale to [0,255]
    img = (img.T/np.amax(f)) * 255
    #Discretize
    img = np.round(img)
    #Convert to int
    img = Image.fromarray(img.astype('uint8')).convert('L')
    #Save it
    img.save(outname + '.png','png')
    return

# This function outputs a .flt
def makeFLT(f, outname):
    #Convert to float32
    img = np.float32(f)
    #Set any negative values to positive machine epsilon
    img = np.maximum(img,np.finfo(np.float32).eps)
    #Save it
    img.tofile(outname + '.flt')
    return

#——————————————————————————————————————————————————————————————————————————————#
# Main                                                                         #
#——————————————————————————————————————————————————————————————————————————————#

#——————————————————————————————#
# Parse Arguments & Initialize #
#——————————————————————————————#
#Get parsed arguments
args = generateParsedArgs()

#Split them up
infile =  args.infile       #input sinogram -- directory or single file
outfolder = args.outfolder  #output directory
x0file = args.x0_file       #initial image (default: zeros)
xtruefile = args.xtrue_file #true image (if available)
psize = args.psize          #pixel size (float) OR file containing pixel sizes
numpix = args.numpix        #size of volume (n x n )
numbin = args.numbin        #number of detector pixels
numtheta = args.numtheta    #number of angles
ns = args.ns                #number of subsets. must divide numtheta evenly
numits = args.num_its       #maximum number of iterations
beta = args.beta            #relaxation parameter beta
epsilon_target = args.epsilon_target #target residual value to stop
theta_range = args.theta_range       #starting and ending angles (deg)
geom = args.geom            #geometry (parallel or fanflat)
dso = args.dso              #source-object distance (cm) (fanbeam only)
dod = args.dod              #detector-object distance (cm) (fanbeam only)
fan_angle = args.fan_angle  #fan angle (deg) (fanbeam only)
make_png = bool(args.make_png)    #whenther or not we will be exporting .png
overwrite = bool(args.overwrite)  #whether we reprocess preexisting files
make_intermediate = bool(args.make_intermediate)  #whether or not you would
                                            #like to generate output each iter

#Were superiorization parameters provided?
use_sup = False #Boolean equal to false 
kmin = 0    #Iteration at which superiorization begins
kstep = 0   #Interval of SARTS between each superiorization step
gamma = 0   #Geometric attenuation factor for superiorization
sigma = 0   #The parameter for BM3D
alpha = 1   #Computed attenuation factor for superiorization, not an arg
if not (args.sup_params is None): #if it is false, then you do not use superiorization 
    use_sup = True #if you are using superiorization. 
    kmin = int(args.sup_params[0])
    kstep = int(args.sup_params[1])
    gamma = args.sup_params[2]
    sigma = args.sup_params[3]

#Get machine epsilon for the float type we are using
eps = np.finfo(float).eps #how accurate you can get with the float type. How small you can get epsilon value. 

#Generate list of filenames from directory provided
fnames = [] #Create the file array
if os.path.isdir(infile):
    fnames = sorted(glob(infile + '/*.flt'))
#Otherwise, a single filename was provided
else:
    fnames.append(infile)

#If pixel size was provided as a floating point value
psizes = 0 #pixel size is irrelevant in flt
try:
    psizes = float(psize)
#Otherwise, a filename was given
except ValueError:
    psizes = np.loadtxt(psize,dtype='f')

#If target residual was provided as a single value
try:
    epsilon_target = float(epsilon_target) 
#Otherwise, a file was provided
except ValueError:
    epsilon_target = np.loadtxt(epsilon_target,dtype='f')

#Create projection geometry
vol_geom = astra.create_vol_geom(numpix, numpix)

#Generate array of angular positions
theta_range = np.deg2rad(theta_range) #convert to radians #0 to 360 becomes 0 to 2pi
angles = theta_range[0] + np.linspace(0,numtheta-1,numtheta,False) \ #how far apart the angles are, where you go by steps of theta. used to create an array of evenly spaced values over a specified range.
         *(theta_range[1]-theta_range[0])/numtheta 

calc_error = False
    
#Create projectors and normalization terms, corresponding to diagonal matrices M and D, for each subset of projection data
#Forming the matrices M and D, because they are inverses of the matrices. Multiplying the diagonal matrix by a vertex
P, Dinv, D_id, Minv, M_id = [None]*ns,[None]*ns,[None]*ns,[None]*ns,[None]*ns #Arrange as None type by the number of subsets arrays, rows which is the number of subsets
for j in range(ns): #iterating over number of subsets 
    ind1 = range(j,numtheta,ns); #Filling these with data, taking steps of ns. numtheta - 1 is the end of the range, 0 - numtheta-1. Where ns is the step size
    p = create_projector(geom,numbin,angles[ind1],dso,dod,fan_angle) #separately defined, sequentially taking slices. 
    
    D_id[j], Dinv[j] = \ #Sequence of arrays D_id(n) = backprojection of the subset matrix n. Submatrix of the whole projection
             astra.create_backprojection(np.ones((numtheta//ns,numbin)),p) #Creating an array of 1's. Number of theta/ns. The numbin is the detector pixels. Parsing projection data and the system matrix. Number of angles/number of subsets. Initialize system matrix to be right dimension of all ones. Put in that matrix and projector geometry and updates. This funciton creates system matrix. Creating backprojection.   
    M_id[j], Minv[j] = \ #M_id are taking in the jth column and the \ move on to the next line. M_id is a matrix of numpix by numpix. 
             astra.create_sino(np.ones((numpix,numpix)),p)  #Retracing, mutiplying, difference sinogram. 
    #Avoid division by zero, also scale M to pixel size
    Dinv[j] = np.maximum(Dinv[j],eps) #What is the use of this line and the following line of code? They have to be bigger than machine epislon, thus you are never dividing by numbers smaller than machine epsilon. 
    Minv[j] = np.maximum(Minv[j],eps) #Whichever is greater
    P[j] = p #jth projection matrix

#Open the file for storing residuals
res_file = open(outfolder + "/residuals.txt", "w+")
res = 0


#————————————#
# Processing #
#————————————#
#For each filename provided
for n in range(len(fnames)):
    #Per-image initialization:
    #Get filename for output
    name = fnames[n]
    head, tail = os.path.split(name)
    #Extract numerical part of filename only. Assumes we have ######_sino.flt
    head, tail = tail.split("_",1)
    outname = outfolder + "/" + head + "_recon_"
    print("\nReconstructing " + head + ":")

    #Read in sinogram
    sino = np.fromfile(name,dtype='f')
    sino = sino.reshape(numtheta,numbin)
    
    #Create a new square nparray for the image size we have
    f = np.zeros((numpix,numpix))

    #Get new psize if they're being read from a file
    try:
        dx = psizes[n] #is the pixel size
    #Otherwise, psize is a float
    except:
        dx = psizes

    #Same for the target residuals
    try:
        etarget = epsilon_target[n]
    except:
        etarget = epsilon_target
        
    #—————————————————————————#
    # Single-image processing #
    #—————————————————————————#
    for k in range(1, numits + 1): #Because you are not starting with 0. 
        #Skip it, if it's built already & we aren't overwriting old ones
        if (not overwrite) and exists(name):
            break
        #——————————————————————————————————————————————————————————————————————#
        # Superiorization step                                                 #
        #——————————————————————————————————————————————————————————————————————#
        if (use_sup) and (k >= kmin) and ((k-kmin)%kstep == 0): #% is mod in python If you are superirizing the image, then k greater than or equal to the k minimum. kstep represents the number of time before next SART iteration
            print("Superiorizing before the next SART iteration...")
            #Apply BM3D
            f_out = bm3d(f,sigma) #the image out is equal to the model 
            #Calc pnorm, what the model does between the images. What the model does to the image. 
            p = f_out - f #The difference between the image, p is an array. 
            pnorm = np.linalg.norm(p,'fro') + eps #The norm of the differences. How bad. Where you add machine eps
            print("pnorm: " + str(pnorm)) #pnorm and the string of pnorm
            #Update alpha
            if k == kmin: #Bottom of the range, first step 
                #Begin with full magnitude of initial transform
                alpha = pnorm #Taking the magnitude of the vector. High norm = higher superiorization step, represents how bad. Each iteration, alpha = pnorm is only for p1
            else:
                #Attenuate for each subsequent superiorization
                alpha *= gamma #Multiplying by a decaying factor, starts at 1
            print("alpha: " + str(alpha) + '\n')
            #Apply alpha if necessary
            if pnorm > alpha: #If norm of the two images, how far apart they are is greater than the computed attenuation factor for superiorization, not an arg. 
                p = alpha * p / (np.linalg.norm(p,'fro') + eps) #Denominator probably doesn't need to be recalculated here, as it's stored in 'pnorm'. Not changing it now, as I'd prefer not to break it by accident.
                f = f + p #Add that difference to the function
            else:
                f = f_out #otherwise just the output
            #Image output
            if make_intermediate:
                makeFLT(f, outname + str(k) + '_bm3d_sup')
                if make_png:
                    makePNG(f, outname + str(k) + '_bm3d_sup')

        #——————————————————————————————————————————————————————————————————————#
        # SART loop                                                            #
        #——————————————————————————————————————————————————————————————————————#
        #Will have to rewrite this SART loop, iterating over number of subsets, as before. 
        for j in range(ns): #Redfine what ns, parser argument. Number of subsets
            ind1 = range(j,numtheta,ns);
            p = P[j] #What is big P, maybe an array? Projection data? P[j] as a vector
            #Forward projection step
            fp_id,fp = astra.create_sino(f,p) #Creating a sinogram, forward projection. Using the image. Creating a sinogram using the image created with superiorization
            #Perform elementwise division
            diffs = (sino[ind1,:] - fp*dx) / Minv[j] / dx #fp is forward project, normalizing data by dividing by the minimum value? Calculates the difference between the sinogram and the forward projection, which is multiplied by the size of the pixels. Which is divided by the matrix, which is then divided by the size of the pixel             
            bp_id,bp = astra.create_backprojection(diffs,p) #Creating the backprojection
            #Get rid of spurious large values
            ind2 = np.abs(bp) > 1e3 #Set anything above a certain range, if its above 1000, it has a value of 0, Boolean, masking 
            bp[ind2] = 0
            #Update f
            f = f + beta * bp / Dinv[j] #what is Dinv?, Dinv is an array. Dinv is the inverse of D matrix. Only index over j because they are diagonal matrix. Beta is the relaxation parameter. bp represents the back projection 
            astra.data2d.delete(fp_id)
            astra.data2d.delete(bp_id)
            
        #——————————————————————————————————————————————————————————————————————#
        # Cleanup                                                              #
        #——————————————————————————————————————————————————————————————————————#
        #Image output
        if make_intermediat
            makeFLT(f, outname + str(k) + '_SART')
            if make_png:
                makePNG(f, outname + str(k) + '_SART')
        #Compute residual
        fp = np.zeros((numtheta,numbin))
        for j in range(ns):
            ind = range(j,numtheta,ns)
            p = P[j]
            fp_tempid,fp_temp = astra.create_sino(f,p)
            fp[ind,:] = fp_temp * dx
            astra.data2d.delete(fp_tempid)
        res = np.linalg.norm(fp-sino,'fro')
        #Error checking
        if calc_error: 
             err = np.linalg.norm(f-xtrue,'fro')/np.linalg.norm(xtrue,'fro')
             print('Iteration #{0:d}: Residual = {1:1.4f}\tError = {2:1.4f}\n'\
                   .format(k,res,err))
        else:
             print('Iteration #{0:d}: Residual = {1:1.4f}\n'.format(k,res))
        #Are we done?
        if (res < etarget):
            print("Target residual for " + head + " of ",end='')
            print(str(etarget) + " reached!")
            break
    
    #——————————————————————————————————————————————————————————————————————————#
    # Single-image Finalization                                                #
    #——————————————————————————————————————————————————————————————————————————#
    #Write the final residual to the file for this image
    res_file.write("%f\n" % res)
    if use_sup:
        makeFLT(f, outname + str(k) + '_BM3Dsup')
        if make_png:
            makePNG(f, outname + str(k) + '_BM3Dsup')
    else:
        makeFLT(f, outname + str(k) + '_SART')
        if make_png:
            makePNG(f, outname + str(k) + '_SART')

#—————————————————————#
# Full Batch Complete #
#—————————————————————#
print("\n\nExiting...")

#Cleanup
for j in range(ns):
    astra.data2d.delete(D_id[j])
    astra.data2d.delete(M_id[j])
    astra.projector.delete(P[j])
res_file.close()

#——————————————————————————————————————————————————————————————————————————————#
# The End!                                                                     #
#——————————————————————————————————————————————————————————————————————————————#






