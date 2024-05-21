import pytest
from pathlib import Path
from detector.model.operations.detector_predictor import ChimpDetectorPredictor

TEST_IM_FILENAME = "small_crystals_2.jpg"
TEST_IM_FILENAME_2 = "big_crystals_3.jpg"
NUM_REAL_IMS = 2
ECHO_MODEL_FILE_NAME = (
    "xchem_chimp_detector_weights.pth"
)
ECHO_NUM_CLASSES = 3


def del_dir(target):
    """
    Delete a given directory and its subdirectories.

    :param target: The directory to delete
    """
    target = Path(target).expanduser()
    assert target.is_dir()
    for p in sorted(target.glob("**/*"), reverse=True):
        if not p.exists():
            continue
        p.chmod(0o666)
        if p.is_dir():
            p.rmdir()
        else:
            p.unlink()
    target.rmdir()


@pytest.fixture()
def cwd():
    return Path(__file__).parent


@pytest.fixture()
def image_folder_real_ims(cwd):
    return cwd / "test_imgs"


@pytest.fixture()
def image_folder_echo_ims(cwd):
    return cwd / "echo_test_imgs"


@pytest.fixture()
def empty_dir(tmp_path):
    tmp_dir = tmp_path / "empty_dir"
    tmp_dir.mkdir(exist_ok=True)
    yield tmp_dir
    del_dir(tmp_dir)


@pytest.fixture()
def echo_detector_model_path(cwd):
    return cwd.parent / f"detector/model/{ECHO_MODEL_FILE_NAME}"


@pytest.fixture()
def echo_detector(echo_detector_model_path, image_folder_echo_ims):
    return ChimpDetectorPredictor(
        echo_detector_model_path,
        list(Path(image_folder_echo_ims).glob("*")),
        ECHO_NUM_CLASSES,
    )
