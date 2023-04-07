params = {
    'data_path': 'data.csv',
    'checkpoint': "AlexWortega/instruct_rugptMedium",
    'additional_special_tokens': [f'speaker{i}: ' for i in range(1, 10)] + ['history: '],
    'num_proc': 5,
    'dataset_name': "under-tree/prepared-yagpt"
}