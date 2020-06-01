import os
from glob import glob
from zipfile import ZipFile

if __name__ == '__main__':
    dataset_dir = '../dataset'
    for dataset_type in ['training', 'test']:
        annos_dir = os.path.join(dataset_dir, dataset_type, 'annotations')
        zip_paths = glob(os.path.join(annos_dir, '*.zip'))
        for zip_idx, zip_path in enumerate(zip_paths):
            print('unzip {}'.format(zip_path))
            zip_fn = os.path.basename(zip_path)[:-4]
            zf = ZipFile(zip_path, 'r')
            zf.extractall(os.path.join(annos_dir, zip_fn))
            zf.close()