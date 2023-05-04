import os
import yaml


from mountaintop import loggerx


class InvalidYAMLException(Exception):
  pass


#####################
## custom tag handler
#####################
def path_join_tagger(loader, node):
    seqs = [str(i) for i in loader.construct_sequence(node)]
    return os.path.join(*seqs)

def line_count_tagger(loader, node):
    filepath = loader.construct_sequence(node)[0]
    line_count = 0
    with open(filepath) as fp:
        line_count = len(fp.readlines())
    return line_count

def load_yaml(filepath):
    # check file exists
    if not os.path.exists(filepath):
        return {}
    
    ## register the tag handler
    yaml.add_constructor('!path_join', path_join_tagger)
    yaml.add_constructor('!line_count', line_count_tagger)
    
    # parse yaml file
    loggerx.info(f"Loading yaml from {filepath}")
    with open(filepath, "r") as fp:
        content = fp.read()
        # docs = yaml.load_all(content, Loader=yaml.FullLoader)
        docs = yaml.load(content, Loader=yaml.FullLoader)
        # docs = yaml.safe_load(fp)
    if not docs:
        raise InvalidYAMLException("Empty yaml file.")
    
    return docs


def save_yaml(filepath, docs):
    loggerx.info(f"Saving yaml to {filepath}")
    with open(filepath, 'w') as fp:
      data = yaml.dump(docs, sort_keys=False)
      fp.write(data)
