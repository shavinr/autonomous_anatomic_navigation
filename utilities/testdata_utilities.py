import os
import glob


def get_data_folder():
    return os.path.dirname(os.path.abspath(__file__)) + "/../../data"


def download_data_webserver(url, save_dir):
    download_config = "--recursive " \
                      "--level=0 " \
                      "--no-parent " \
                      "--no-host-directories " \
                      "--cut-dirs=3 " \
                      "--no-verbose " \
                      "--reject='index.html*' " \
                      "--directory-prefix={save_dir} " \
                      "--timestamping " \
                      "--user=simulus " \
                      "--password=testdata".format(save_dir=save_dir)
    download_command = "wget {config} {url}".format(config=download_config, url=url)
    ret = os.system(download_command)
    if ret > 0:
        raise Warning("Test data download unsuccessful. Please check if wget works.")
    else:
        assert os.path.exists(save_dir), "Test data does not exist."


def download_example_data(save_dir=None):
    if save_dir is None:
        save_dir = get_data_folder()

    url = "folk.ntnu.no/androst/data/simulus/"
    if not os.path.exists(save_dir):
        download_data_webserver(url, save_dir)

    # wget on windows does not clean up index files.
    files_to_remove = glob.glob(os.path.join(get_data_folder(), "**/index*"), recursive=True)
    [os.remove(f) for f in files_to_remove]
