params = {
    'data_path': 'encoded_texts.pkl',
    'checkpoint': "AlexWortega/instruct_rugptMedium",
    'additional_special_tokens': [f'speaker{i}: ' for i in range(1, 10)] + ['history: '],
    'num_proc': 5
}