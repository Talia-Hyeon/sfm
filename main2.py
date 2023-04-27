from match import *
from sfm import *


def read_files(path):
    txt_file = osp.join(path, 'K.txt')
    K = np.loadtxt(txt_file)
    img_files = [osp.join(path, file) for file in os.listdir(path) if file.endswith('.JPG')]
    img_files.sort()
    return K, img_files


if __name__ == '__main__':
    img_root = './data/'
    K, img_files = read_files(img_root)
    view_l = create_view(img_files)
    all_matches = create_all_matches(view_l)
    points_3D = reconstruct(K, view_l, all_matches)
