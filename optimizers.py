class AdversarialOptimizerSimultaneousWithExtra(object):
	"""
	Perform simultaneous updates for each player in the game, with an additional optimizer/param/loss
	set. This is the easiest way to accomplish adding an additional "loss" to a single model, without
	modifying larger sections of code.
	There's probably a better way to do this, but this is simple enough and works
	"""
	def __init__(self, extra_losses, extra_params, extra_optimizers):
		self.extra_losses = extra_losses
		self.extra_params = extra_params
		self.extra_optimizers = extra_optimizers

	def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,function_kwargs):
        return K.function(inputs,
                          outputs,
                          updates=self.call(losses, params, optimizers, constraints) + model_updates,
                          **function_kwargs)

    def call(self, losses, params, optimizers, constraints):
        updates = []
        for loss, param, optimizer, constraint in zip(losses, params, optimizers, constraints):
            updates += optimizer.get_updates(param, constraint, loss)
        for loss, param, optimizer in zip(self.extra_losses, self.extra_params, self.extra_optimizers):
        	updates += optimizer.get_updates(param, {}, loss)
        return updates
