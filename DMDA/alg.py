from DMDA.DMDA import DMDA

algorithms = {
    'DMDA',
}

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
                'algorithm not found: {}'.format(algorithm_name))
    return globals()[algorithm_name]