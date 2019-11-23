import os
import string

def rename_data():
    # Character data from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    character_order = string.digits + string.ascii_uppercase + string.ascii_lowercase
    print(character_order)
    dir_names = sorted(os.listdir('data'))
    print(len(character_order), len(dir_names))
    assert len(dir_names) == len(character_order)

    for c, dir_name in zip(character_order, dir_names):
        if dir_name.startswith('Sample'):
            old_path = os.path.join('data', dir_name)
            if c in string.digits:
                os.rename(old_path, os.path.join('data', c + '_digit'))
            elif c in string.ascii_uppercase:
                os.rename(old_path, os.path.join('data', c + '_upper'))
            if c in string.ascii_lowercase:
                os.rename(os.path.join('data', dir_name), os.path.join('data', c + '_lower'))

if __name__ == '__main__':
    rename_data()