def calculate(config, callback=lambda img, i: None):
    evaluator = config.mod.Evaluator(config)
    for i in range(1, config.num_iterations+1):
        print('Start of iteration', i)
        x = evaluator.eval_and_train()
        img = x.copy()
        img = config.mod.Evaluator.postprocess_image(img)
        callback(img, i)
        if i == config.num_iterations:
            return img
