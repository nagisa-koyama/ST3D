def get_ontology_mapping(input_ontology, output_ontology):
    # KITTI ontology
    # https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    # Waymo ontology
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md
    # Lyft ontology
    # https://wandb.ai/wandb/lyft/reports/An-Exploration-of-Lyft-s-Self-Driving-Car-Dataset--Vmlldzo0MzcyNw#the-9-classes-in-the-lyft-dataset
    # Pandaset ontology
    # https://github.com/scaleapi/pandaset-devkit/blob/master/docs/annotation_instructions_cuboids.pdf
    # Nuscenes ontology
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md
    classes_kitti = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare']
    classes_waymo = ['Vehicle', 'Pedestrian', 'Cyclist', 'Sign']
    classes_lyft = ['car', 'pedestrian', 'truck', 'bicycle', 'motorcycle',
                    'bus', 'emergency_vehicle', 'other_vehicle', 'animal']
    classes_pandaset = ['Car', 'Pickup Truck', 'Medium-sized Truck', 'Semi-truck', 'Towed Object', 'Motorcycle', 'Other Vehicle - Construction Vehicle', 'Other Vehicle - Uncommon', 'Other Vehicle - Pedicab', 'Emergency Vehicle', 'Bus', 'Personal Mobility Device',
                        'Motorized Scooter', 'Bicycle', 'Train', 'Trolley', 'Tram / Subway', 'Pedestrian', 'Pedestrian with Object', 'Animals - Bird', 'Animals - Other', 'Pylons', 'Road Barriers', 'Signs', 'Cones', 'Construction Signs', 'Temporary Construction Barriers', 'Rolling Containers']
    classes_nuscenes = ['car', 'truck', 'bus', 'construction_vehicle', 'motorcycle',
                        'bicycle', 'trailer', 'pedestrian', 'traffic_cone', 'barrier', 'ignore']
    map_lyft_to_kitti = {
        'car': 'Car',
        'pedestrian': 'Pedestrian',
        'truck': 'Truck',
        'bicycle': 'Cyclist',
        'motorcycle': 'Cyclist',
        'bus': 'Car',
        'emergency_vehicle': 'Misc',
        'other_vehicle': 'Misc',
        'animal': 'Misc'
    }
    map_waymo_to_kitti = {
        'Vehicle': 'Car',
        'Pedestrian': 'Pedestrian',
        'Cyclist': 'Cyclist',
        'Sign': 'Misc'
    }
    map_pandaset_to_kitti = {
        'Car': 'Car',
        'Pickup Truck': 'Truck',
        'Medium-sized Truck': 'Truck',
        'Semi-truck': 'Truck',
        'Towed Object': 'Misc',
        'Motorcycle': 'Cyclist',
        'Other Vehicle - Construction Vehicle': 'Misc',
        'Other Vehicle - Uncommon': 'Misc',
        'Other Vehicle - Pedicab': 'Misc',
        'Emergency Vehicle': 'Misc',
        'Bus': 'Car',
        'Personal Mobility Device': 'Misc',
        'Motorized Scooter': 'Cyclist',
        'Bicycle': 'Cyclist',
        'Train': 'Tram',
        'Trolley': 'Tram',
        'Tram / Subway': 'Tram',
        'Pedestrian': 'Pedestrian',
        'Pedestrian with Object': 'Pedestrian',
        'Animals - Bird': 'Misc',
        'Animals - Other': 'Misc',
        'Pylons': 'Misc',
        'Road Barriers': 'Misc',
        'Signs': 'Misc',
        'Cones': 'Misc',
        'Construction Signs': 'Misc',
        'Temporary Construction Barriers': 'Misc',
        'Rolling Containers': 'Misc'
    }
    map_nuscenes_to_kitti = {
        'car': 'Car',
        'truck': 'Truck',
        'bus': 'Car',
        'construction_vehicle': 'Misc',
        'motorcycle': 'Cyclist',
        'bicycle': 'Cyclist',
        'trailer': 'Misc',
        'pedestrian': 'Pedestrian',
        'traffic_cone': 'Misc',
        'barrier': 'Misc',
        'ignore': 'Misc'
        # Follwoing classes are defined in the intruction but not in the dataset.
        # 'Bicycle Rack': 'Misc',
        # 'Police Vehicle': 'Misc',
        # 'Ambulance': 'Misc',
        # 'Child Pedestrian': 'Pedestrian',
        # 'Construction Worker': 'Pedestrian',
        # 'Stroller': 'Misc',
        # 'Wheelchair': 'Misc',
        # 'Portable Personal Mobility Vehicle': 'Misc',
        # 'Police Officer': 'Pedestrian',
        # 'Animal': 'Misc',
        # 'Pushable Pullable Object': 'Misc',
        # 'Debris': 'Misc'
    }
    map_nuscenes_to_waymo = {
        'car': 'Vehicle',
        'truck': 'Vehicle',
        'bus': 'Vehicle',
        'construction_vehicle': 'Sign',  # 'Sign' as a catch-all for all other objects
        'motorcycle': 'Sign',
        'bicycle': 'Cyclist',
        'trailer': 'Sign',
        'pedestrian': 'Pedestrian',
        'traffic_cone': 'Sign',
        'barrier': 'Sign',
        'ignore': 'Sign'
    }
    map_nuscenes_to_lyft = {
        'car': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'construction_vehicle': 'other_vehicle',
        'motorcycle': 'motorcycle',
        'bicycle': 'bicycle',
        'trailer': 'other_vehicle',
        'pedestrian': 'pedestrian',
        'traffic_cone': 'animal',  # 'animal' as a catch-all for all other objects
        'barrier': 'animal',
        'ignore': 'animal',
    }
    map_nuscenes_to_pandaset = {
        'car': 'Car',
        'truck': 'Medium-sized Truck',
        'bus': 'Bus',
        'construction_vehicle': 'Other Vehicle - Construction Vehicle',
        'motorcycle': 'Motorcycle',
        'bicycle': 'Bicycle',
        'trailer': 'Towed Object',
        'pedestrian': 'Pedestrian',
        'traffic_cone': 'Cones',
        'barrier': 'Temporary Construction Barriers',
        'ignore': 'Animals - Other',  # 'Animals - Other' as a catch-all for all other objects
    }
    map_kitti_to_lyft = {
        'Car': 'car',
        'Pedestrian': 'pedestrian',
        'Truck': 'truck',
        'Cyclist': 'bicycle',
        'Van': 'other_vehicle',
        'Misc': 'other_vehicle',
        'Person_sitting': 'pedestrian',
        'Tram': 'other_vehicle',
        'Misc': 'other_vehicle',
        'DontCare': 'other_vehicle',
    }
    map_kitti_to_waymo = {
        'Car': 'Vehicle',
        'Pedestrian': 'Pedestrian',
        'Truck': 'Vehicle',
        'Cyclist': 'Cyclist',
        'Van': 'Vehicle',
        'Misc': 'Sign',  # 'Sign' as a catch-all for all other objects
        'Person_sitting': 'Pedestrian',
        'Tram': 'Sign',
        'Misc': 'Sign',
        'DontCare': 'Sign',
    }
    map_kitti_to_pandaset = {
        'Car': 'Car',
        'Pedestrian': 'Pedestrian',
        'Truck': 'Medium-sized Truck',
        'Cyclist': 'Bicycle',
        'Van': 'Medium-sized Truck',
        'Misc': 'Sign',  # 'Sign' as a catch-all for all other objects
        'Person_sitting': 'Pedestrian',
        'Tram': 'Sign',
        'Misc': 'Sign',
        'DontCare': 'Sign',
    }
    map_kitti_to_nuscenes = {
        'Car': 'car',
        'Pedestrian': 'pedestrian',
        'Truck': 'truck',
        'Cyclist': 'bicycle',
        'Van': 'car',
        'Misc': 'debris',  # 'Debris as a catch-all for all other objects
        'Person_sitting': 'pedestrian',
        'Tram': 'debris',
        'Misc': 'debris',
        'DontCare': 'debris',
    }
    map_lyft_to_nuscenes = {
        'car': 'car',
        'pedestrian': 'pedestrian',
        'truck': 'truck',
        'bicycle': 'bicycle',
        'motorcycle': 'motorcycle',
        'bus': 'bus',
        'emergency_vehicle': 'ignore',
        'other_vehicle': 'ignore',
        'animal': 'ignore'
    }
    map_waymo_to_nuscenes = {
        'Vehicle': 'car',
        'Pedestrian': 'pedestrian',
        'Cyclist': 'bicycle',
        'Sign': 'ignore'
    }
    map_pandaset_to_nuscenes = {
        'Car': 'car',
        'Pickup Truck': 'truck',
        'Medium-sized Truck': 'truck',
        'Semi-truck': 'truck',
        'Towed Object': 'trailer',
        'Motorcycle': 'motorcycle',
        'Other Vehicle - Construction Vehicle': 'construction_vehicle',
        'Other Vehicle - Uncommon': 'ignore',
        'Other Vehicle - Pedicab': 'ignore',
        'Emergency Vehicle': 'ignore',
        'Bus': 'bus',
        'Personal Mobility Device': 'ignore',
        'Motorized Scooter': 'bicycle',
        'Bicycle': 'bicycle',
        'Train': 'ignore',
        'Trolley': 'ignore',
        'Tram / Subway': 'ignore',
        'Pedestrian': 'pedestrian',
        'Pedestrian with Object': 'pedestrian',
        'Animals - Bird': 'ignore',
        'Animals - Other': 'ignore',
        'Pylons': 'traffic_cone',
        'Road Barriers': 'barrier',
        'Signs': 'ignore',
        'Cones': 'traffic_cone',
        'Construction Signs': 'ignore',
        'Temporary Construction Barriers': 'barrier',
        'Rolling Containers': 'ignore'
    }

    map_head_per_dataset_to_kitti = {
        'kitti:Car': 'kitti:Car',
        'waymo:Vehicle': 'kitti:Car',
        'lyft:car': 'kitti:Car',
        'pandaset:Car': 'kitti:Car',
        'nuscenes:car': 'kitti:Car',
        'kitti:Pedestrian': 'kitti:Pedestrian',
        'waymo:Pedestrian': 'kitti:Pedestrian',
        'lyft:pedestrian': 'kitti:Pedestrian',
        'pandaset:Pedestrian': 'kitti:Pedestrian',
        'nuscenes:pedestrian': 'kitti:Pedestrian'
    }
    map_head_per_dataset_to_nuscenes = {
        'kitti:Car': 'nuscenes:car',
        'waymo:Vehicle': 'nuscenes:car',
        'lyft:car': 'nuscenes:car',
        'pandaset:Car': 'nuscenes:car',
        'nuscenes:car': 'nuscenes:car',
        'kitti:Pedestrian': 'nuscenes:pedestrian',
        'waymo:Pedestrian': 'nuscenes:pedestrian',
        'lyft:pedestrian': 'nuscenes:pedestrian',
        'pandaset:Pedestrian': 'nuscenes:pedestrian',
        'nuscenes:pedestrian': 'nuscenes:pedestrian'
    }
    map_head_per_dataset_to_lyft = {
        'kitti:Car': 'lyft:car',
        'waymo:Vehicle': 'lyft:car',
        'lyft:car': 'lyft:car',
        'pandaset:Car': 'lyft:car',
        'nuscenes:car': 'lyft:car',
        'kitti:Pedestrian': 'lyft:pedestrian',
        'waymo:Pedestrian': 'lyft:pedestrian',
        'lyft:pedestrian': 'lyft:pedestrian',
        'pandaset:Pedestrian': 'lyft:pedestrian',
        'nuscenes:pedestrian': 'lyft:pedestrian',
    }
    # Defined for compatibility with the rest of the code but not used.
    map_kitti_to_head_per_dataset = {
        'Car': 'waymo:Vehicle',
        'Pedestrian': 'waymo:Pedestrian',
        'Truck': 'waymo:Vehicle',
        'Cyclist': 'waymo:Cyclist',
        'Van': 'waymo:Vehicle',
        'Misc': 'waymo:Sign',
        'Person_sitting': 'waymo:Pedestrian',
        'Tram': 'waymo:Sign',
        'Misc': 'waymo:Sign',
        'DontCare': 'waymo:Sign',
    }
    # Defined for compatibility with the rest of the code but not used.
    map_nuscenes_to_head_per_dataset = {
        'nuscenes:car': 'waymo:Vehicle',
        'nuscenes:truck': 'waymo:Vehicle',
        'nuscenes:bus': 'waymo:Vehicle',
        'nuscenes:construction_vehicle': 'waymo:Vehicle',
        'nuscenes:motorcycle': 'waymo:Cyclist',
        'nuscenes:bicycle': 'waymo:Cyclist',
        'nuscenes:trailer': 'waymo:Sign',
        'nuscenes:pedestrian': 'waymo:Pedestrian',
        'nuscenes:traffic_cone': 'waymo:Sign',
        'nuscenes:barrier': 'waymo:Sign',
        'nuscenes:ignore': 'waymo:Sign'
    }
    # Defined for compatibility with the rest of the code but not used.
    map_lyft_to_head_per_dataset = {
        'car': 'waymo:Vehicle',
        'pedestrian': 'waymo:Pedestrian',
        'truck': 'waymo:Vehicle',
        'bicycle': 'waymo:Cyclist',
        'motorcycle': 'waymo:Cyclist',
        'bus': 'waymo:Vehicle',
        'emergency_vehicle': 'waymo:Sign',
        'other_vehicle': 'waymo:Sign',
        'animal': 'waymo:Sign'
    }
    # Supports identical mapping
    if input_ontology == output_ontology:
        map_identical = {}
        classes = []
        if input_ontology == 'kitti':
            classes = classes_kitti
        elif input_ontology == 'waymo':
            classes = classes_waymo
        elif input_ontology == 'lyft':
            classes = classes_lyft
        elif input_ontology == 'pandaset':
            classes = classes_pandaset
        elif input_ontology == 'nuscenes':
            classes = classes_nuscenes
        else:
            assert False, input_ontology + ' to ' + output_ontology + ' is not supported'
        for cls in classes:
            map_identical[cls] = cls
        return map_identical
    elif input_ontology == 'lyft' and output_ontology == 'kitti':
        return map_lyft_to_kitti
    elif input_ontology == 'waymo' and output_ontology == 'kitti':
        return map_waymo_to_kitti
    elif input_ontology == 'pandaset' and output_ontology == 'kitti':
        return map_pandaset_to_kitti
    elif input_ontology == 'nuscenes' and output_ontology == 'kitti':
        return map_nuscenes_to_kitti
    elif input_ontology == 'nuscenes' and output_ontology == 'waymo':
        return map_nuscenes_to_waymo
    elif input_ontology == 'nuscenes' and output_ontology == 'lyft':
        return map_nuscenes_to_lyft
    elif input_ontology == 'nuscenes' and output_ontology == 'pandaset':
        return map_nuscenes_to_pandaset
    elif input_ontology == 'kitti' and output_ontology == 'lyft':
        return map_kitti_to_lyft
    elif input_ontology == 'kitti' and output_ontology == 'waymo':
        return map_kitti_to_waymo
    elif input_ontology == 'kitti' and output_ontology == 'pandaset':
        return map_kitti_to_pandaset
    elif input_ontology == 'kitti' and output_ontology == 'nuscenes':
        return map_kitti_to_nuscenes
    elif input_ontology == 'waymo' and output_ontology == 'nuscenes':
        return map_waymo_to_nuscenes
    elif input_ontology == 'lyft' and output_ontology == 'nuscenes':
        return map_lyft_to_nuscenes
    elif input_ontology == 'pandaset' and output_ontology == 'nuscenes':
        return map_pandaset_to_nuscenes
    elif input_ontology == 'head_per_dataset' and output_ontology == 'kitti':
        return map_head_per_dataset_to_kitti
    elif input_ontology == 'kitti' and output_ontology == 'head_per_dataset':
        return map_kitti_to_head_per_dataset
    elif input_ontology == 'head_per_dataset' and output_ontology == 'nuscenes':
        return map_head_per_dataset_to_nuscenes
    elif input_ontology == 'nuscenes' and output_ontology == 'head_per_dataset':
        return map_nuscenes_to_head_per_dataset
    elif input_ontology == 'head_per_dataset' and output_ontology == 'lyft':
        return map_head_per_dataset_to_lyft
    elif input_ontology == 'lyft' and output_ontology == 'head_per_dataset':
        return map_lyft_to_head_per_dataset
    else:
        assert False, input_ontology + ' to ' + output_ontology + ' is not supported'
