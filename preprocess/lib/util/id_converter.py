class IDConverter(object):

    def __init__(self):
        self._id2name = []
        self._name2id = {}

    def to_name(self, id_):
        if len(self._id2name) <= id_:
            raise ValueError('Invalid index')
        return self._id2name[id_]

    def to_id(self, name):
        if name not in self._name2id:
            self._name2id[name] = len(self._name2id)
            self._id2name.append(name)
        return self._name2id[name]

    @property
    def id2name(self):
        return self._id2name

    @property
    def name2id(self):
        return self._name2id

    @property
    def unique_num(self):
        return len(self._id2name)
