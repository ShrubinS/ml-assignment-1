
def iterate_into_out_dict(scores, algorithm, name, out_dict, chunk_size):
    for score_name, score_value in scores.iteritems():
        key_name = algorithm + ";" + name + ";" + score_name
        if key_name in out_dict:
            chunk_data = out_dict[key_name]
        else:
            chunk_data = {}
            out_dict[key_name] = chunk_data
        chunk_data[chunk_size] = score_value
