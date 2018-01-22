

def detect(config):

    if config["library"] == "tensorflow":
        from inference.tf_op import graph_prep, run_detection
        detection_graph, label_map, categories, category_index = graph_prep(config["model"]["classes"],
            config["model"]["model_path"],config["model"]["pbtxt"])

        run_detection(config["device_path"], detection_graph, label_map, categories, category_index, 
            config["show_stream"], config["show_stream"], config["write_output"], config["benchmark"])
    
    elif config["library"] == "movidius":
        from inference.mvnc_op import test        
    