from keras.models import Model
from keras.layers import Flatten, Concatenate
from keras.applications import vgg16


class PerceptualModel(Model):

	def __init__(self, img_size):
		super().__init__()

		vgg = vgg16.VGG16(include_top=False, input_shape=(img_size[0], img_size[1], 3))
		layer_ids = [2, 5, 9, 13, 17]
		layer_outputs = [Flatten()(vgg.layers[layer_id].output) for layer_id in layer_ids]
		features = Concatenate(axis=-1)(layer_outputs)

		self.__model = Model(inputs=vgg.input, outputs=features)

	def call(self, inputs, mask=None):
		return self.__model(vgg16.preprocess_input(inputs))

	def compute_output_shape(self, input_shape):
		return self.__model.compute_output_shape(input_shape)
