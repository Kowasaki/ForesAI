

def detect(config):

    if config["library"] == "tensorflow":
        from inference.tf_op import load_model, load_split_model, load_label_map, run_detection

        score = None
        expand = None
        if config["model"]["split_hack"]:
            detection_graph, score, expand = load_split_model(config["model"]["model_path"])
        else:
            detection_graph = load_model(config["model"]["model_path"])

        label_map, categories, category_index = load_label_map(config["model"]["classes"],
            config["model"]["pbtxt"])

        run_detection(config["device_path"], detection_graph, label_map, categories, category_index, 
            config["show_stream"], config["show_stream"], config["write_output"], 
            config["ros_enabled"], config["benchmark"], score_node = score, expand_node = expand)
    
    elif config["library"] == "movidius":
        from inference.mvnc_op import run_detection
        run_detection(config["device_path"],config["model"]["model_path"], config["show_stream"], 
            config["ros_enabled"])        
    