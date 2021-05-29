from dataset.JsonFromFiles import JsonFromFilesDataset
from .RTEDataset import RTEDataset
from .SST2Dataset import SST2Dataset
from .WikiREDataset import WikiREDataset
dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "RTE": RTEDataset,
    "SST2": SST2Dataset,
    "RE": WikiREDataset,
}
