from torch.utils.data import Dataset 
from libraries.strategies import * 

class Source(Dataset):
	def __init__(self, root, mapper):
		super(Source, self).__init__()
		self.root = root 
		self.mapper = mapper 
		self.image_paths = pull_files(self.root, '*.jpg')[:30000]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		image = read_image(image_path=self.image_paths[index], by='th')
		return self.mapper(image)