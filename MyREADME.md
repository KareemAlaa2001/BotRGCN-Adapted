## TODO

# using this as a playground for my thoughts on work

### Aspects of the pipeline that can be varied:

#### Graph Construction

- All tweets augmented vs a subset
- Homogeneous vs Heterogeneous graph
- Label rate of user set maintained within the user nodes in the graph

#### Hyperparameters: 

embedding_size, dropout,learning rate,weight_decay, svdComponents

#### Model Construction:
- Sizes of num_prop, cat_prop and des_embedding with respect to the embedding size

- Inclusion and size of the additional layer between the concatenated components' MLP results and the GCN model (as well as a counterpart for tweets (or whether the tweets actually get 2 in this case))

#### Dataset Construction (smaller priority)

- Number of as well as which features are included in num_prop and cat_prop
- do we include a feature for tweets on the number of mentions?
- do we include average number of mentions per tweet for each user?


### More TODOs from the supervisor meeting 01/07:

Explore directionality of edges
Explore which types to keep and which to drop
Is the directionality & typing redundant? Might make sense to make the graph undirected
Read up carefully on directionality in the employed models
Replacing RGCN with something else that supports full hetero

Area under the precision-recall curve (rather than ROC curve)
Explore class imbalance
