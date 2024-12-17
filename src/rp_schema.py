INPUT_SCHEMA = {
    'modifier_token': {
        'type': str,
        'required': True
    },
    'id': {
        'type': str,
        'required': True,
        'default': None
    },
    'training_id': {
        'type': str,
        'required': True,

    },
    's3_url': {
        'type': str,
        'required': True,
        'default': None
    }}
