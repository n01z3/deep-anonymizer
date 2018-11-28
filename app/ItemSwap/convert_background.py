import os
from .scripts_background.options.test_options import TestOptions
from .scripts_background.data import CreateDataLoader
from .scripts_background.models import create_model
from .scripts_background.util.visualizer import save_images_to_folder


def run(opt):
    # hard-code some parameters for test
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_inputA(data)
        model.testA()
        visuals = model.get_visuals_fakeB()
        img_path = model.get_image_paths()
        save_images_to_folder(opt.results_dir, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                              width=opt.display_winsize)


if __name__ == '__main__':
    opt = TestOptions().parse()
    run(opt)
