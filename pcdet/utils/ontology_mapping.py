def get_ontology_mapping(input_ontology, output_ontology):
    # KITTI ontology
    # https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    # Waymo ontology
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md
    # Lyft ontology
    # https://wandb.ai/wandb/lyft/reports/An-Exploration-of-Lyft-s-Self-Driving-Car-Dataset--Vmlldzo0MzcyNw#the-9-classes-in-the-lyft-dataset
    # Pandaset ontology 
    # https://github.com/scaleapi/pandaset-devkit/blob/master/docs/annotation_instructions_cuboids.pdf
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
        'Misc': 'Sign',
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
        'Misc': '',
        'Person_sitting': 'Pedestrian',
        'Tram': 'Sign',
        'Misc': 'Sign',
        'DontCare': 'Sign',
    }
    map_head_per_dataset_to_kitti = {
        'kitti:Car': 'kitti:Car',
        'waymo:Vehicle': 'kitti:Car',
        'lyft:car': 'kitti:Car',
        'pandaset:Car': 'kitti:Car',
        'kitti:Pedestrian': 'kitti:Pedestrian',
        'waymo:Pedestrian': 'kitti:Pedestrian',
        'lyft:pedestrian': 'kitti:Pedestrian',
        'pandaset:Pedestrian': 'kitti:Pedestrian',
    }
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

    if input_ontology == 'lyft' and output_ontology == 'kitti':
        return map_lyft_to_kitti
    elif input_ontology == 'waymo' and output_ontology == 'kitti':
        return map_waymo_to_kitti
    elif input_ontology == 'pandaset' and output_ontology == 'kitti':
        return map_pandaset_to_kitti
    elif input_ontology == 'kitti' and output_ontology == 'lyft':
        return map_kitti_to_lyft
    elif input_ontology == 'kitti' and output_ontology == 'waymo':
        return map_kitti_to_waymo
    elif input_ontology == 'kitti' and output_ontology == 'pandaset':
        return map_kitti_to_pandaset
    elif input_ontology == 'head_per_dataset' and output_ontology == 'kitti':
        return map_head_per_dataset_to_kitti
    elif input_ontology == 'kitti' and output_ontology == 'head_per_dataset':
        return map_kitti_to_head_per_dataset
    else:
        assert False, input_ontology + ' to ' + output_ontology + ' is not supported'