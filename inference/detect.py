from inference.inference_builder import InferenceBuilder

def old_detect(config):
    """
    The old detect function. Takes in a config file and determines which deep learning library 
    and task to perform

    config: config json with all the necessary intructions
    """

    if config["library"] == "tensorflow":
        from inference.ops.tf_op import load_model, load_split_model, load_label_map, run_detection

        score = None
        expand = None
        if config["model"]["split_hack"]:
            detection_graph, score, expand = load_split_model(config["model"]["model_path"])
        else:
            detection_graph = load_model(config["model"]["model_path"])

        label_map, categories, category_index = load_label_map(config["model"]["classes"],
            config["model"]["pbtxt"])
        
        if config["model"]["mask_enabled"]:
            from inference.ops.tf_op import run_mask_detection
            run_mask_detection(config["device_path"], detection_graph, label_map, categories, category_index, 
                config["show_stream"], config["show_stream"], config["write_output"], 
                config["ros_enabled"], config["benchmark"], graph_trace_enabled = config["model"]["graph_trace"],
                score_node = score, expand_node = expand)
        else:
            run_detection(config["device_path"], detection_graph, label_map, categories, category_index, 
                config["show_stream"], config["show_stream"], config["write_output"], 
                config["ros_enabled"], config["benchmark"], graph_trace_enabled = config["model"]["graph_trace"],
                score_node = score, expand_node = expand)
    
    elif config["library"] == "movidius":
        from inference.ops.mvnc_op import run_detection
        run_detection(config["device_path"],config["model"]["model_path"], config["show_stream"], 
            config["ros_enabled"])        

    elif config["library"] == "pytorch":
        from inference.ops.pytorch_op import run_detection
        run_detection(config["device_path"],
            config["model"]["model_path"],
            config["model"]["model_name"],
            config["model"]["weights_path"],
            config["model"]["classes"],
            show_window = config["show_stream"],
            visualize = config["show_stream"], 
            write_output = config["write_output"],
            ros_enabled = config["ros_enabled"], 
            usage_check = config["benchmark"])
    else:
        raise Exception("Library not supported!")

def detect(config):
    """
    Detect takes in a config file and determines which deep learning library and task to perform
    config: config json with all the necessary intructions
    
    Currently, the old detect is called for libraries where the loader class hasn't been completed yet
    """
    if config["library"] == "pytorch" or config["library"] == "darknet" :
        inference = InferenceBuilder(config)
        inference.detect()
    else:
        old_detect(config)