import gzip
import os, shutil, wget
# STOLEN from NeMo


def prepare_arpa_file(arpa):
    lm_gzip_path = f'{arpa}.gz'
    if not os.path.exists(lm_gzip_path) and not os.path.isfile(arpa):
        assert arpa in ["3-gram.arpa","4-gram.arpa","3-gram.pruned.1e-7.arpa","3-gram.pruned.3e-7.arpa"]
        print('Downloading pruned 3-gram model.')
        lm_url = f'http://www.openslr.org/resources/11/{lm_gzip_path}'
        lm_gzip_path = wget.download(lm_url)
        print(f'Downloaded {lm_gzip_path}')
    else:
        print('Pruned .arpa.gz already exists.')
    uppercase_lm_path = f'{arpa}'
    if not os.path.exists(uppercase_lm_path):
        with gzip.open(lm_gzip_path, 'rb') as f_zipped:
            with open(uppercase_lm_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
        print('Unzipped the 3-gram language model.')
    else:
        print('Unzipped .arpa already exists.')
    lm_path = f'lowercase_{arpa}'
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    print('Converted language model file to lowercase.')
    return lm_path

if __name__ == '__main__':
    arpa = '3-gram.pruned.1e-7.arpa'
    prepare_arpa_file(arpa)