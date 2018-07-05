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

	cur_stage = [True, output_shape]
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
	shapes.append([False])
	shapes = [[not shapes[i+1][0],*x[1:][::-1] ] for i, x in enumerate(shapes[:-1])][::-1]
	shapes[0].pop(0)

	return shapes

## Pretty formatting!
## Get the architecture from back_calc_nice, preferably (similar structure).
def format_unet_nice( unet_arch ):
	mat = [[None]]
	extend_up = lambda : mat.insert(0, [None]*len(mat[0])) ## borks when used
	extend_right = lambda : [m.append(None) for m in mat]
	extend_down = lambda : mat.append([None]*len(mat[0]))

	posi, posj, midx = 0, 0, 0
	for stage in unet_arch:
		if type(stage[0])==bool:
			if stage[0]:
				posi += 1
			else:
				posi -= 1
			stage = stage[1:]
			midx += 1
		for val in stage:
			if posi < 0:
				extend_up()
			if posi >= len(mat):
				extend_down()
			if posj >= len(mat[0]):
				extend_right()
			mat[posi][posj] = (val, midx)
			midx += 1
			posj += 1
		posj -= 1
	omat = mat
	mat = [[str(x[0]) if x else "" for x in y] for y in mat]
	maxwidth = max([max([len(x) for x in y]) for y in mat])
	output_str = ["" for _ in mat]
	for i,row in enumerate(mat):
		for j,val in enumerate(row):
			if j > 0:
				if row[j-1] and val:
					output_str[i] += " -> "
				else:
					output_str[i] += "    "
			output_str[i] += val + " "*(maxwidth-len(val))
	final_output = ""
	for i,row in enumerate(mat):
		final_output += output_str[i] + "\n"
		if i == len(mat)-1:
			continue
		nline1, nline2 = "", ""
		for j in range(len(row)):
			if mat[i][j] and mat[i+1][j]:
				if omat[i+1][j][1] > omat[i][j][1]:
					nline1 += "|" + " "*(maxwidth-1)
					nline2 += "v" + " "*(maxwidth-1)
				else:
					nline1 += "^" + " "*(maxwidth-1)
					nline2 += "|" + " "*(maxwidth-1)
			else:
				nline1 += " "*maxwidth
				nline2 += " "*maxwidth
			nline1 += "    "
			nline2 += "    "
		final_output += nline1 + "\n" + nline2 + "\n"
	return final_output #"\n".join(output_str)