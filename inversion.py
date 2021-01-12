import argparse
import pickle
import os
import imageio
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import dnnlib
import dnnlib.tflib as tflib
import config

from perceptual_model import PerceptualModel

STYLEGAN_MODEL_URL = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'


def optimize_latent_codes(args):
	tflib.init_tf()

	with dnnlib.util.open_url(STYLEGAN_MODEL_URL, cache_dir=config.cache_dir) as f:
		_G, _D, Gs = pickle.load(f)

	latent_code = tf.get_variable(
		name='latent_code', shape=(1, 18, 512), dtype='float32', initializer=tf.initializers.zeros()
	)

	generated_img = Gs.components.synthesis.get_output_for(latent_code, randomize_noise=False)
	generated_img = tf.transpose(generated_img, [0, 2, 3, 1])
	generated_img = ((generated_img + 1) / 2) * 255
	generated_img_for_display = tf.saturate_cast(generated_img, tf.uint8)

	target_img = tf.placeholder(tf.float32, [None, args.input_img_size[0], args.input_img_size[1], 3])

	target_img_resized = tf.image.resize_images(
		target_img, tuple(args.perceptual_img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
	)

	generated_img_resized = tf.image.resize_images(
		generated_img, tuple(args.perceptual_img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
	)

	perceptual_model = PerceptualModel(img_size=args.perceptual_img_size)
	generated_img_features = perceptual_model(generated_img_resized)
	target_img_features = perceptual_model(target_img_resized)

	loss_op = tf.reduce_mean(tf.abs(generated_img_features - target_img_features))

	if args.optimizer == 'adam':
		optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
	else:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)

	train_op = optimizer.minimize(loss_op, var_list=[latent_code])

	sess = tf.get_default_session()

	img_names = sorted(os.listdir(args.imgs_dir))
	for img_name in img_names:
		img = imageio.imread(os.path.join(args.imgs_dir, img_name))

		sess.run(tf.variables_initializer([latent_code] + optimizer.variables()))

		progress_bar_iterator = tqdm(
			iterable=range(args.total_iterations),
			bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
			desc=img_name
		)

		for i in progress_bar_iterator:
			loss, _ = sess.run(
				fetches=[loss_op, train_op],
				feed_dict={
					target_img: img[np.newaxis, ...]
				}
			)

			progress_bar_iterator.set_postfix_str('loss=%.2f' % loss)

		reconstructed_imgs, latent_codes = sess.run(
			fetches=[generated_img_for_display, latent_code],
			feed_dict={
				target_img: img[np.newaxis, ...]
			}
		)

		imageio.imwrite(os.path.join(args.reconstructions_dir, img_name), reconstructed_imgs[0])
		np.savez(file=os.path.join(args.latents_dir, img_name + '.npz'), latent_code=latent_codes[0])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--imgs-dir', type=str, required=True)
	parser.add_argument('--reconstructions-dir', type=str, required=True)
	parser.add_argument('--latents-dir', type=str, required=True)

	parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
	parser.add_argument('--input-img-size', type=int, nargs=2, default=(1024, 1024))
	parser.add_argument('--perceptual-img-size', type=int, nargs=2, default=(256, 256))
	parser.add_argument('--learning-rate', type=float, default=1e-3)
	parser.add_argument('--total-iterations', type=int, default=1000)

	args = parser.parse_args()

	os.makedirs(args.reconstructions_dir, exist_ok=True)
	os.makedirs(args.latents_dir, exist_ok=True)

	optimize_latent_codes(args)
