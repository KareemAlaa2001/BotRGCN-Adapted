from crossValTrainTest import trainValModelForCrossVal

import numpy as np
import wandb

if __name__ == '__main__':
    # Current Values
    config_defaults = dict(
        model_name="TweetAugHANConfigurable",
        dataset="HeteroTwibot",
        embedding_size = 204,
        dropout = 0.18380768518137663,
        lr = 0.004164987490510339,
        weight_decay = 0.0027187218127487783,
        svdComponents = 100,
        thirds = False,
        epochs = 40,
        extraLayer = True,
        numHANLayers = 4,
        neighboursPerNode = 382,
        batch_size = 256,
        # neighboursPerNode = 10,
        # batch_size=1,
        testing_enabled = False,
        crossValFolds = 5
    )

    wandb.init(project="test-project", entity="graphbois",  config=config_defaults)

    config = wandb.config

    aggregate_results = {}

    for i in range(config.crossValFolds):
        val_results = trainValModelForCrossVal(config.embedding_size, config.dropout, config.lr, \
            config.weight_decay, config.svdComponents, config.thirds, config.epochs, config.extraLayer, \
                config.numHANLayers, config.neighboursPerNode, config.batch_size, config.testing_enabled, \
                    using_external_config=True, augmentedDataset=False, datasetVariant=1, crossValFolds=5, \
                        crossValIteration=i, dev=False)
        
        for key in val_results:
            if key not in aggregate_results:
                aggregate_results[key] = []
            
            if key != 'conf_matrix':
                aggregate_results[key].append(val_results[key])
            else:
                aggregate_results[key].append(val_results[key].numpy())

    mean_results = {}
    result_stdev = {}
    print(aggregate_results)
    for key in aggregate_results:
        mean_results["mean_" + key] = np.array(aggregate_results[key]).mean(axis=0)
        result_stdev["stdev_" + key] = np.array(aggregate_results[key]).std(axis=0)

    wandb.log(mean_results)
    wandb.log(result_stdev)
