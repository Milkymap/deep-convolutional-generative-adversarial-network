import click 
import pickle 

import cv2

import torch as th 
import torch.nn as nn 
import torch.optim as optim 


from torchvision import transforms as T 
from torch.utils.data import DataLoader 

from libraries.strategies import * 
from libraries.log import logger 
from machine_learning.producer import Source
from machine_learning.structure import Generator, Descriminator 


@click.command()
@click.option('--root', help='path to source data', required=True)
@click.option('--z_dim', help='dimension of latent vector', default=64)
@click.option('--fmap_dim', help='dimension of features map', default=64)
@click.option('--desc_in_channel', help='number input channels (descriminator)', default=3)
@click.option('--nb_epochs', help='number of epochs', default=16)
@click.option('--batch_size', help='size of batch data', default=32)
@click.option('--cv_display/--no-cv_display', default=True)
def main_loop(root, z_dim, fmap_dim, desc_in_channel, nb_epochs, batch_size, cv_display):
	device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
	logger.info(f'models run on {device}')

	descriminator = Descriminator(in_channel=desc_in_channel, fmap_dim=fmap_dim).to(device)
	generator = Generator(z_dim=z_dim, fmap_dim=fmap_dim, out_channel=desc_in_channel).to(device)

	mapper = create_image_mapper((fmap_dim, fmap_dim))
	source = Source(root, mapper)
	loader = DataLoader(dataset=source, shuffle=True, batch_size=batch_size, num_workers=2)

	criterion = nn.BCELoss()
	descriminator_solver = optim.Adam(descriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
	generator_solver = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
	distribution = th.randn((64, z_dim, 1, 1))
	accumulator = []
	message = '[%03d/%03d]:%05d | E_D : %07.3f | E_G : %07.3f'
	counter = 0
	while counter < nb_epochs:
		for idx, input_batch in enumerate(loader):
			X = input_batch.to(device)
			RL = th.ones(X.shape[0], device=device).float()
			FL = th.zeros(X.shape[0], device=device).float()
			# train descriminator 
			descriminator_solver.zero_grad()
			Dx = descriminator(X)
			
			E_Dx = criterion(Dx.view(-1), RL) 
			
			Z = th.randn((X.shape[0], z_dim, 1, 1), device=device)
			Gz = generator(Z)
			D_Gz = descriminator(Gz.detach())
			E_DGz = criterion(D_Gz.view(-1), FL)
			
			E_Dx_DGz = (E_Dx + E_DGz) / 2
			
			E_Dx_DGz.backward()
			descriminator_solver.step()

			#train generator
			generator_solver.zero_grad()
			D_Gz = descriminator(Gz)
			E_DGz = criterion(D_Gz.view(-1), RL)
			E_DGz.backward()
			generator_solver.step()

			logger.debug( message % (counter, nb_epochs, idx, E_Dx_DGz.item(), E_DGz.item()) )

			with th.no_grad():
				response = generator(distribution).detach().cpu()
				grid_images = to_grid(response, normalize=True)
				grid_images = th2cv(grid_images)
				if cv_display:
					cv2.imshow('000', grid_images)
					cv2.waitKey(5)
				if idx % 16 == 0:
					accumulator.append(grid_images)

		counter = counter + 1
	# end loop 
	
	#th.save(generator, 'storage/generator.pt')
	#th.save(descriminator, 'storage/descriminator.pt')
	with open('storage/timelapse.pkl', 'wb') as fp:
		pickle.dump(accumulator, fp)

if __name__ == '__main__':
	main_loop()
	

