## Collection of random tools/utilities for size/shape calculation

## Back calculate the input shape from provided output shape, shape decrease per
## "stage", and up/down sampling factor for a U-Net
def back_calc( output_shape, decrease_per=4, factor=2, stages=3 ):
	shapes = [output_shape]
	is_nice = True
	for i in range(stages):
		shapes.append(shapes[-1] + decrease_per)
		is_nice &= (shapes[-1]%factor==0)
		shapes.append(shapes[-1]//factor)
	for i in range(stages):
		shapes.append(shapes[-1] + decrease_per)
		shapes.append(shapes[-1]*factor)
	shapes.append(shapes[-1] + decrease_per)
	return (shapes, is_nice)

## Returns a format-friendly version of the result from back_calc
## i.e. something that can be put into a ascii-rendering method relatively easily
def back_calc_nice( output_shape, decrease_per=2, num_times=2, factor=2, stages=3 ):
	last = lambda l : l[-1][-1]

	cur_stage = [output_shape]
	shapes = [cur_stage]

	for _ in range(stages):
		for _ in range(num_times):
			cur_stage.append(last(shapes) + decrease_per)
		cur_stage = [True, last(shapes)//factor]
		shapes.append(cur_stage)
	for _ in range(stages+1):
		for _ in range(num_times):
			cur_stage.append(last(shapes) + decrease_per)
		cur_stage = [False, last(shapes)*factor]
		shapes.append(cur_stage)
	shapes.pop()

	return shapes

## Pretty formatting!
## Get the architecture from back_calc_nice, preferably (similar structure).
def format_unet_nice( unet_arch ):
	mat = [[None]]
	extend_up = lambda : mat.insert([None]*len(mat[0]))
	extend_right = lambda : [m.append(None) for m in mat]
	extend_down = lambda : mat.append([None]*len(mat[0]))
	## WIP