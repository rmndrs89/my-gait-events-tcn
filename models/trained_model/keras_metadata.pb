
�*root"_tf_keras_network*�*{"name": "tcn_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "tcn_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputs"}, "name": "inputs", "inbound_nodes": []}, {"class_name": "TCN", "config": {"name": "tcn_layer", "trainable": true, "dtype": "float32", "nb_filters": 16, "kernel_size": 5, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "same", "use_skip_connections": true, "dropout_rate": 0.0, "return_sequences": true, "activation": "relu", "use_batch_norm": true, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "he_normal"}, "name": "tcn_layer", "inbound_nodes": [[["inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "initial_contact", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "initial_contact", "inbound_nodes": [[["tcn_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "final_contact", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final_contact", "inbound_nodes": [[["tcn_layer", 0, 0, {}]]]}], "input_layers": [["inputs", 0, 0]], "output_layers": [["initial_contact", 0, 0], ["final_contact", 0, 0]]}, "shared_object_id": 8, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 6]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 6]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 6]}, "float32", "inputs"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 6]}, "float32", "inputs"]}, "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "tcn_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputs"}, "name": "inputs", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "TCN", "config": {"name": "tcn_layer", "trainable": true, "dtype": "float32", "nb_filters": 16, "kernel_size": 5, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "same", "use_skip_connections": true, "dropout_rate": 0.0, "return_sequences": true, "activation": "relu", "use_batch_norm": true, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "he_normal"}, "name": "tcn_layer", "inbound_nodes": [[["inputs", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "initial_contact", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "initial_contact", "inbound_nodes": [[["tcn_layer", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "final_contact", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final_contact", "inbound_nodes": [[["tcn_layer", 0, 0, {}]]], "shared_object_id": 7}], "input_layers": [["inputs", 0, 0]], "output_layers": [["initial_contact", 0, 0], ["final_contact", 0, 0]]}}, "training_config": {"loss": {"initial_contact": {"class_name": "MyWeightedMeanSquaredError", "config": {"reduction": "auto", "name": null}, "shared_object_id": 10}, "final_contact": {"class_name": "MyWeightedMeanSquaredError", "config": {"reduction": "auto", "name": null}, "shared_object_id": 10}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "inputs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputs"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "tcn_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "TCN", "config": {"name": "tcn_layer", "trainable": true, "dtype": "float32", "nb_filters": 16, "kernel_size": 5, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "same", "use_skip_connections": true, "dropout_rate": 0.0, "return_sequences": true, "activation": "relu", "use_batch_norm": true, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "he_normal"}, "inbound_nodes": [[["inputs", 0, 0, {}]]], "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 6]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "initial_contact", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "initial_contact", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tcn_layer", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "final_contact", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "final_contact", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tcn_layer", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�*root.layer_with_weights-0.residual_block_0"_tf_keras_layer*�{"name": "residual_block_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 6]}}2
�*root.layer_with_weights-0.residual_block_1"_tf_keras_layer*�{"name": "residual_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�*root.layer_with_weights-0.residual_block_2"_tf_keras_layer*�{"name": "residual_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�
&root.layer_with_weights-0.slicer_layer"_tf_keras_layer*�	{"name": "Slice_Output", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "Slice_Output", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAEwAAAHMYAAAAfABkAGQAhQKIAGoAZABkAIUCZgMZAFMAqQFO\nKQHaEm91dHB1dF9zbGljZV9pbmRleCkB2gJ0dKkB2gRzZWxmqQD6VC9ob21lL3JvYmJpbi9Qcm9q\nZWN0cy9teS1nYWl0LWV2ZW50cy10Y24vdmVudi9saWIvcHl0aG9uMy44L3NpdGUtcGFja2FnZXMv\ndGNuL3Rjbi5wedoIPGxhbWJkYT41AQAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [{"class_name": "TCN", "config": {"name": "tcn_layer", "trainable": true, "dtype": "float32", "nb_filters": 16, "kernel_size": 5, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "same", "use_skip_connections": true, "dropout_rate": 0.0, "return_sequences": true, "activation": "relu", "use_batch_norm": true, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "he_normal"}}]}]}, "function_type": "lambda", "module": "tcn.tcn", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 13}2
�	_;root.layer_with_weights-0.residual_block_0.shape_match_conv"_tf_keras_layer*�{"name": "matching_conv1D", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "matching_conv1D", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 6}}, "shared_object_id": 17}}2
�`;root.layer_with_weights-0.residual_block_0.final_activation"_tf_keras_layer*�{"name": "Act_Res_Block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Res_Block", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 18}2
�	a3root.layer_with_weights-0.residual_block_0.conv1D_0"_tf_keras_layer*�{"name": "conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv1D_0", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 6}}, "shared_object_id": 22}}2
�b>root.layer_with_weights-0.residual_block_0.batch_normalization"_tf_keras_layer*�{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}, "shared_object_id": 28}}2
�c7root.layer_with_weights-0.residual_block_0.Act_Conv1D_0"_tf_keras_layer*�{"name": "Act_Conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv1D_0", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 29}2
�d5root.layer_with_weights-0.residual_block_0.SDropout_0"_tf_keras_layer*�{"name": "SDropout_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SpatialDropout1D", "config": {"name": "SDropout_0", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�	e3root.layer_with_weights-0.residual_block_0.conv1D_1"_tf_keras_layer*�{"name": "conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv1D_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}, "shared_object_id": 35}}2
�f@root.layer_with_weights-0.residual_block_0.batch_normalization_1"_tf_keras_layer*�{"name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 37}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 39}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}, "shared_object_id": 41}}2
�g7root.layer_with_weights-0.residual_block_0.Act_Conv1D_1"_tf_keras_layer*�{"name": "Act_Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv1D_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 42}2
�h5root.layer_with_weights-0.residual_block_0.SDropout_1"_tf_keras_layer*�{"name": "SDropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SpatialDropout1D", "config": {"name": "SDropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�i:root.layer_with_weights-0.residual_block_0.Act_Conv_Blocks"_tf_keras_layer*�{"name": "Act_Conv_Blocks", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv_Blocks", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 45}2
�q;root.layer_with_weights-0.residual_block_1.shape_match_conv"_tf_keras_layer*�{"name": "matching_identity", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "matching_identity", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAABAAAAUwAAAHMEAAAAfABTAKkBTqkAKQHaAXhyAgAAAHICAAAA+lQv\naG9tZS9yb2JiaW4vUHJvamVjdHMvbXktZ2FpdC1ldmVudHMtdGNuL3ZlbnYvbGliL3B5dGhvbjMu\nOC9zaXRlLXBhY2thZ2VzL3Rjbi90Y24ucHnaCDxsYW1iZGE+hgAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "tcn.tcn", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 46}2
�r;root.layer_with_weights-0.residual_block_1.final_activation"_tf_keras_layer*�{"name": "Act_Res_Block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Res_Block", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 47}2
�	s3root.layer_with_weights-0.residual_block_1.conv1D_0"_tf_keras_layer*�{"name": "conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv1D_0", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}, "shared_object_id": 51}}2
�t@root.layer_with_weights-0.residual_block_1.batch_normalization_2"_tf_keras_layer*�{"name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 53}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 55}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 56, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}, "shared_object_id": 57}}2
�u7root.layer_with_weights-0.residual_block_1.Act_Conv1D_0"_tf_keras_layer*�{"name": "Act_Conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv1D_0", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 58}2
�v5root.layer_with_weights-0.residual_block_1.SDropout_0"_tf_keras_layer*�{"name": "SDropout_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SpatialDropout1D", "config": {"name": "SDropout_0", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�	w3root.layer_with_weights-0.residual_block_1.conv1D_1"_tf_keras_layer*�{"name": "conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv1D_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}, "shared_object_id": 64}}2
�x@root.layer_with_weights-0.residual_block_1.batch_normalization_3"_tf_keras_layer*�{"name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 66}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 67}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 68}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 69, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}, "shared_object_id": 70}}2
�y7root.layer_with_weights-0.residual_block_1.Act_Conv1D_1"_tf_keras_layer*�{"name": "Act_Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv1D_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 71}2
�z5root.layer_with_weights-0.residual_block_1.SDropout_1"_tf_keras_layer*�{"name": "SDropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SpatialDropout1D", "config": {"name": "SDropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 72, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�{:root.layer_with_weights-0.residual_block_1.Act_Conv_Blocks"_tf_keras_layer*�{"name": "Act_Conv_Blocks", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv_Blocks", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 74}2
��;root.layer_with_weights-0.residual_block_2.shape_match_conv"_tf_keras_layer*�{"name": "matching_identity", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "matching_identity", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAABAAAAUwAAAHMEAAAAfABTAKkBTqkAKQHaAXhyAgAAAHICAAAA+lQv\naG9tZS9yb2JiaW4vUHJvamVjdHMvbXktZ2FpdC1ldmVudHMtdGNuL3ZlbnYvbGliL3B5dGhvbjMu\nOC9zaXRlLXBhY2thZ2VzL3Rjbi90Y24ucHnaCDxsYW1iZGE+hgAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "tcn.tcn", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 75}2
��;root.layer_with_weights-0.residual_block_2.final_activation"_tf_keras_layer*�{"name": "Act_Res_Block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Res_Block", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 76}2
�	�3root.layer_with_weights-0.residual_block_2.conv1D_0"_tf_keras_layer*�{"name": "conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv1D_0", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [4]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 77}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 78}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 79, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}, "shared_object_id": 80}}2
��@root.layer_with_weights-0.residual_block_2.batch_normalization_4"_tf_keras_layer*�{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 81}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 82}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 83}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 84}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 85, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}, "shared_object_id": 86}}2
��7root.layer_with_weights-0.residual_block_2.Act_Conv1D_0"_tf_keras_layer*�{"name": "Act_Conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv1D_0", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 87}2
��5root.layer_with_weights-0.residual_block_2.SDropout_0"_tf_keras_layer*�{"name": "SDropout_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SpatialDropout1D", "config": {"name": "SDropout_0", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 88, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 89}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
�	�3root.layer_with_weights-0.residual_block_2.conv1D_1"_tf_keras_layer*�{"name": "conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv1D_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [4]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 90}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 91}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 92, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}, "shared_object_id": 93}}2
��@root.layer_with_weights-0.residual_block_2.batch_normalization_5"_tf_keras_layer*�{"name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 94}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 95}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 96}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 97}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 98, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}, "shared_object_id": 99}}2
��7root.layer_with_weights-0.residual_block_2.Act_Conv1D_1"_tf_keras_layer*�{"name": "Act_Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv1D_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 100}2
��5root.layer_with_weights-0.residual_block_2.SDropout_1"_tf_keras_layer*�{"name": "SDropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SpatialDropout1D", "config": {"name": "SDropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 101, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 102}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 16]}}2
��:root.layer_with_weights-0.residual_block_2.Act_Conv_Blocks"_tf_keras_layer*�{"name": "Act_Conv_Blocks", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "Act_Conv_Blocks", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 103}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 104}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "Mean", "name": "initial_contact_loss", "dtype": "float32", "config": {"name": "initial_contact_loss", "dtype": "float32"}, "shared_object_id": 105}2
��root.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "Mean", "name": "final_contact_loss", "dtype": "float32", "config": {"name": "final_contact_loss", "dtype": "float32"}, "shared_object_id": 106}2