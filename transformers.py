import os
import subprocess


class FaceSwapper:
    def __init__(self):
        self.DESTINATION = 'images/face_swap'

    def transform(self, filepath):
        rel_filepath = os.path.join('..', filepath)
        destination_filepath = os.path.join(self.DESTINATION, os.path.basename(filepath))
        os.makedirs(self.DESTINATION, exist_ok=True)
        subprocess.check_call(['bash', 'face_swap.sh', rel_filepath,
                               os.path.join('..', destination_filepath)], shell=False)
        return destination_filepath


if __name__ == '__main__':
    swapper = FaceSwapper()
    swapper.transform('images/upload/DSC_0696.jpg')
