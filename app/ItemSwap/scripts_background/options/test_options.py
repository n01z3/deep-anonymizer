from .base_options import BaseOptions


ALL_OPTIONS = [
    'dataroot',
    'batch_size',
   'loadSize',
    'fineSize',
    'display_winsize',
    'input_nc',
    'output_nc',
    'ngf',
    'ndf',
    'netD',
    'netG',
    'n_layers_D',
    'gpu_ids',
    'name',
    'dataset_mode',
    'model',
    'direction',
    'epoch',
    'load_iter',
    'num_threads',
    'checkpoints_dir',
    'norm',
    'serial_batches',
    'no_dropout',
    'max_dataset_size',
    'resize_or_crop',
    'no_flip',
    'init_type',
    'init_gain',
    'verbose',
    'suffix'

    'ntest',
    'results_dir',
    'aspect_ratio',
    'phase',
    #  Dropout and Batchnorm has different behavioir during training and test.
    'eval',
    'num_test',
    'isTrain'
]

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
