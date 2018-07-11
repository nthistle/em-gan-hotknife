




def get_masked_loss(batch_size, output_shape, mask_size, slice_index):

	mask = np.zeros((batch_size,) + output_shape + (1,), dtype=np.float32)
	slices = [slice()]*3

	lower_bound = (output_shape[slice_index] - mask_size)//2
	upper_bound = (output_shape[slice_index] + mask_size)//2

	# set lower to 1
	slices[slice_index] = slice(None,lower_bound)
	mask[:, slices[0], slices[1], slices[2], :] = 1.0

	# set upper to 1
	slices[slice_index] = slice(upper_bound,None)
	mask[:, slices[0], slices[1], slices[2], :] = 1.0

	def masked_loss(y_true, y_pred):
		y_true_masked = tf.multiply(y_true, mask)
		y_pred_masked = tf.multiply(y_pred, mask)
		return mean_squared_error(y_true_masked, y_pred_masked)

	return masked_loss



def train(generator, discriminator, generator_optimizer, discriminator_optimizer, penalty_optimizer,
	epochs, minibatch_size, num_minibatch, instance_noise, instance_noise_std_dev, input_shape, output_shape,
	generator_mask_size):

	discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

	generator.name = "pretrained_generator"
	generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

	penalty_z = Input(shape=input_shape+(1,))
	penalty = Model(penalty_z, generator(penalty_z))
	penalty.compile(loss=)


def train(generator_filename, epochs=25, batch_size=64, num_batches=32,
	disc_lr=1e-7, gen_lr=1e-6, penalty_lr=1e-5, num_passive=2, noise=0.0,
	disc_optim="adam", gen_optim="adam", pen_optim="adam", batch_norm=False,
	lower_noise=False, disc_regularization=0, sc_enabled=False,
	data_file="hotknifedata.hdf5", output_folder="run_output"):

	disc_optim = disc_optim.lower()
	gen_optim = gen_optim.lower()
	pen_optim = pen_optim.lower()

	print("Running em-hotknife GAN training for %d epochs with parameters:" % epochs)
	print("Discriminator LR: %f" % disc_lr)
	print("Generator LR: %f" % gen_lr)
	print("Penalty LR: %f" % penalty_lr)
	print("Outputting data to %s" % output_folder)

	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	discriminator_optimizer = None
	if disc_optim == "adam":
		discriminator_optimizer = Adam(disc_lr)
	elif disc_optim == "sgd":
		discriminator_optimizer = SGD(disc_lr)

	generator_optimizer = None
	if gen_optim == "adam":
		generator_optimizer = Adam(gen_lr)
	elif gen_optim == "sgd":
		generator_optimizer = SGD(gen_lr)

	penalty_optimizer = None
	if pen_optim == "adam":
		penalty_optimizer = Adam(penalty_lr)
	elif pen_optim == "sgd":
		penalty_optimizer = SGD(penalty_lr)

	discriminator = get_discriminator(batch_norm=batch_norm, regularization=disc_regularization)
	discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

	generator = load_model(generator_filename)
	generator.name = "pretrained_generator"
	generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer) # Fairly certain this doesn't get directly used anywhere

	penalty_z = Input(shape=(64,64,64,1))
	penalty = Model(penalty_z, generator(penalty_z))
	penalty.compile(loss=get_masked_loss(batch_size), optimizer=penalty_optimizer)

	z = Input(shape=(64,64,64,1))
	img = generator(z)

	discriminator.trainable = False

	valid = discriminator(img)

	combined = Model(z, valid)
	combined.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

	# for sampling the data for training
	fake_gen = h5_gap_data_generator_valid(data_file,"volumes/data", (64,64,64), batch_size)

	# for training discriminator on what is real
	real_gen = h5_nogap_data_generator(data_file,"volumes/data", (32,32,32), batch_size)

	# just for periodically sampling the generator to see what's going on
	test_gen = h5_gap_data_generator_valid(data_file,"volumes/data", (64,64,64), 5)
	common_test_gen = h5_gap_data_generator_valid(data_file, "volumes/data", (64,64,64), 15)

	common_test = common_test_gen.__next__()  ## Will sample this same one every epoch to better see what it's learning


	## Once we've had 3+ "critical" epochs (where discriminator loss is much lower than generator loss), terminate training early, if short circuiting enabled
	CRITICAL_EPOCH_THRESH = 3
	critical_epochs = 0

	history = {"epoch":[], "d_loss":[], "d_acc":[], "g_loss":[], "g_penalty":[]}

	## Just do an "Epoch 0" test
	latent_samp = fake_gen.__next__()
	gen_output = generator.predict(latent_samp)
	real_data = real_gen.__next__()
	d_loss = 0.5 * np.add(discriminator.test_on_batch(real_data, np.ones((batch_size, 1))), discriminator.test_on_batch(gen_output, np.zeros((batch_size, 1))))
	g_loss = combined.test_on_batch(latent_samp, np.ones((batch_size, 1)))
	g_loss_penalty = penalty.test_on_batch(latent_samp, get_center_of_valid_block(latent_samp))
	print("Epoch #0 [D loss: %f acc: %f] [G loss: %f penalty: %f]" % (d_loss[0], d_loss[1], g_loss, g_loss_penalty))
	print("="*50)

	history["epoch"].append(0)
	history["d_loss"].append(d_loss[0])
	history["d_acc"].append(d_loss[1])
	history["g_loss"].append(g_loss)
	history["g_penalty"].append(g_loss_penalty)
	## End our Epoch 0 stuff

	current_noise = noise

	for epoch in range(epochs):

		if lower_noise:
			current_noise = max(min(noise * (1 - 2*(epoch-(epochs/4))/epochs),1),0)
		
		g_loss = None
		g_loss_penalty = None
		d_loss = None

		for n in range(num_batches): # do n minibatches

			# train discriminator
			latent_samp = fake_gen.__next__() # input to generator
			gen_output = generator.predict(latent_samp)

			if current_noise > 0.0:
				gen_output = apply_noise(gen_output, current_noise) # instance noise, sorta
			# hopefully helps a little bit?

			real_data = real_gen.__next__()

			d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
			d_loss_fake = discriminator.train_on_batch(gen_output, np.zeros((batch_size, 1)))
			d_loss_new = (1./num_batches) * 0.5 * np.add(d_loss_real, d_loss_fake)

			if d_loss is None:
				d_loss = d_loss_new
			else:
				d_loss = np.add(d_loss, d_loss_new)

			# train generator
			latent_samp = fake_gen.__next__()

			if epoch < num_passive:
				g_loss_new = (1./num_batches) * combined.test_on_batch(latent_samp, np.ones((batch_size, 1)))
			else:
				g_loss_new = (1./num_batches) * combined.train_on_batch(latent_samp, np.ones((batch_size, 1)))

			## Now penalty instead of generator
			if epoch < num_passive:
				g_loss_penalty_new = (1./num_batches) * penalty.test_on_batch(latent_samp, get_center_of_valid_block(latent_samp))
			else:
				g_loss_penalty_new = (1./num_batches) * penalty.train_on_batch(latent_samp, get_center_of_valid_block(latent_samp))

			if g_loss is None:
				g_loss = g_loss_new
				g_loss_penalty = g_loss_penalty_new
			else:
				g_loss = np.add(g_loss, g_loss_new)
				g_loss_penalty = np.add(g_loss_penalty, g_loss_penalty_new)

		print("Epoch #%d [D loss: %f acc: %f] [G loss: %f penalty: %f]" % (epoch+1, d_loss[0], d_loss[1], g_loss, g_loss_penalty))

		if epoch > num_passive + CRITICAL_EPOCH_THRESH and (g_loss/d_loss[0]) > 2.0:
			critical_epochs += 1
		else:
			critical_epochs = 0

		# now save some sample input
		#prev = test_gen.__next__()

		#outp = generator.predict(prev)
		common_outp = generator.predict(common_test)

		#if noise > 0.0:
		#	outp = apply_noise(outp, noise)
		#	common_outp = apply_noise(common_outp, noise)

		#write_sampled_output_even(prev, outp, os.path.join(output_folder,"train_epoch_%03d.png"%(epoch+1)))
		write_sampled_output_even_large(common_test, common_outp, os.path.join(output_folder,"test_epoch_%03d.png"%(epoch+1)), 32)

		if (epoch+1)%5 == 0:
			generator.save(os.path.join(output_folder,"generator_train_epoch_%03d.h5"%(epoch+1)))
			discriminator.save(os.path.join(output_folder,"discriminator_train_epoch_%03d.h5"%(epoch+1)))

		# Update History
		history["epoch"].append(epoch+1)
		history["d_loss"].append(d_loss[0])
		history["d_acc"].append(d_loss[1])
		history["g_loss"].append(g_loss)
		history["g_penalty"].append(g_loss_penalty)

		if critical_epochs > CRITICAL_EPOCH_THRESH and sc_enabled:
			print("[INFO] SHORT CIRCUITING TRAINING! g_loss > 2*d_loss for 3+ epochs!")
			break

	generator.save(os.path.join(output_folder,"generator_train_final.h5"))
	discriminator.save(os.path.join(output_folder,"discriminator_train_final.h5"))

	with open(os.path.join(output_folder,"history.csv"),"w") as f:
		pandas.DataFrame(history).reindex(columns=["epoch","d_loss","d_acc","g_loss","g_penalty"]).to_csv(f, index=False)



if __name__=="__main__":
	main()