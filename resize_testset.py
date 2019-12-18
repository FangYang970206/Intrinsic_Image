import os
import scipy.misc as misc


def main():
    source_path = 'F:\\BOLD\\test'
    target_path = 'F:\\BOLD\\resize_test'
    selections = ['orig', 'refl', 'sha']

    if not os.path.exists(target_path):
        os.mkdir(target_path)
        os.mkdir(os.path.join(target_path, 'orig'))
        os.mkdir(os.path.join(target_path, 'refl'))
        os.mkdir(os.path.join(target_path, 'sha'))

    for sel in selections:
        img_names = os.listdir(os.path.join(source_path, sel))
        for img_name in img_names:
            img = misc.imread(os.path.join(os.path.join(source_path, sel), img_name))
            img = misc.imresize(img, (256, 256))
            misc.imsave(os.path.join(os.path.join(target_path, sel), img_name), img)


if __name__ == '__main__':
    main()
