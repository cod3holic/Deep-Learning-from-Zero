def _dot_var(v, verbose=False):
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ':'
        name += str(v.shape) + ' ' + str(v.dtype)
    return f"{id(v)} [label='{name}', color=orange, style=filled]\n"

def _dot_func(f):
    txt = f"{id(f)} [label='{f.__class__.__name__}']\n"
    
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    seen_set = set()
    funcs = []

    def add_func(f):
        if f not in seen_set:
            seen_set.add(f)
            funcs.append(f)
    
    add_func(output.creator)
    txt += _dot_var(output)

    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose=verbose)
            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'