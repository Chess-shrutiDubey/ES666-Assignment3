import pdb
import src
import glob
import importlib.util
import os
import cv2

### Change path to images here
# Update the path to point to your Images directory correctly
path = '/home/shruti/assignment-03/ES666-Assignment3/Images/*'  # Full path to the images
###

all_submissions = glob.glob('./src/*')
os.makedirs('./results/', exist_ok=True)

for idx, algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(algo.split(os.sep)[-1], idx + 1, len(all_submissions)))
    try:
        module_name = '{}_{}'.format(algo.split(os.sep)[-1], 'stitcher')
        filepath = '{}{}stitcher.py'.format(algo, os.sep, 'stitcher.py')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        ### Process each sub-directory in the Images path
        for impaths in glob.glob(path):
            print('\t\t Processing... {}'.format(impaths))
            stitched_image, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths)

            if stitched_image is not None:
                # Create a valid output filename
                outfile = './results/{}/{}.png'.format(impaths.split(os.sep)[-1], module_name)
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                cv2.imwrite(outfile, stitched_image)
                print(homography_matrix_list)
                print('Panorama saved ... @ {}'.format(outfile))
            else:
                print("No stitched image to save.")

            print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')
